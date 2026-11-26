from settings import *
from weight_selection import fit_incremental,select_truncation
from weight_validation import STATUS,require_unblinding_authorization,write_atomic
import json
import numpy as np
import pandas as pd
import duckdb

def capped(raw,stabilized,bounds):
    return raw*np.minimum(1,10/np.maximum(stabilized,1e-15)) if "stabilized_cap" in bounds else np.clip(raw,*bounds["raw"])

def freeze(db,data):
    b,h,p,f,design,chosen=data
    fit=chosen["fit"]
    rows=[]
    for arm in (0,1):
        original=fit.hourly if arm==0 else fit.full_liberal
        terminal=original.groupby("cohort_id").tail(1).set_index("cohort_id")
        raw=terminal.raw.reindex(b.cohort_id).fillna(1).to_numpy()
        sw=terminal.stabilized.reindex(b.cohort_id).fillna(1).to_numpy()
        bounds=fit.bounds[arm]
        end=capped(raw,sw,bounds)
        gate=fit.controls if arm==0 else fit.intervention
        if chosen["calibration"]!="none":
            flag=f.category[chosen["calibration"]].eq("True").set_axis(b.cohort_id)
            q=gate.copy()
            q["flag"]=flag.loc[q.cohort_id].to_numpy()
            q["factor"]=q.raw_p01_p99/capped(q.raw,q.stabilized,bounds)
            factors=q.groupby("flag").factor.mean()
            require(len(factors)==2 and factors.between(0.7,1.4).all(),"Invalid frozen calibration factor")
            end*=flag.loc[b.cohort_id].map(factors).to_numpy()
        require(np.allclose(pd.Series(end,index=b.cohort_id).loc[gate.cohort_id],gate.raw_p01_p99),
                "Frozen weights differ from the blinded pass calculation")
        frame=pd.DataFrame({"cohort_id":b.cohort_id,"arm":arm,"w_after_g":end})
        for j in range(6):
            if arm==0 and j:
                lookup=original.loc[original.hour_idx.lt(j)].groupby("cohort_id").tail(1).set_index("cohort_id").raw
                frame[f"w_start_{j}"]=lookup.reindex(b.cohort_id).fillna(1).to_numpy()
            else:
                frame[f"w_start_{j}"]=1.0
        rows.append(frame)
    out=pd.concat(rows,ignore_index=True)
    file=WEIGHTS/f"{db}_locked_main.parquet"
    temp=file.with_suffix(".partial.parquet")
    with duckdb.connect(config={"temp_directory":str(RUNTIME)}) as con:
        con.register("frozen",out)
        con.execute("COPY frozen TO ? (FORMAT PARQUET,COMPRESSION ZSTD)",[str(temp)])
    temp.replace(file)
    return digest(file)

def run_lock():
    for path in (RUNTIME, TABLES, WEIGHTS):
        path.mkdir(parents=True, exist_ok=True)
    STATUS.unlink(missing_ok=True)
    protected=protected_digests()
    out,attempts,state=fit_incremental()
    d={db:x[2] for db,x in out.items()}
    passed=all(s["passes"] for s in d.values())
    require(protected==protected_digests(),"cohort and cloning inputs changed")
    summary=pd.concat([v[0] for v in out.values()],ignore_index=True)
    ess=pd.concat([v[1] for v in out.values()],ignore_index=True)
    for arm in (0,1):
        n=summary.database.map({db:v[f"arm{arm}_balance"] for db,v in d.items()})
        col=f"unweighted_arm{arm}_count"
        m=f"unweighted_arm{arm}_mean"
        x=pd.to_numeric(summary[col],errors="coerce")
        mask=summary[col].eq("<10")|((n-x>0)&(n-x<10))
        summary[m]=summary[m].astype(object)
        summary.loc[mask,m]="Masked"
        summary.loc[mask & summary[col].ne("<10"),col]="Masked"
    summary.to_csv(TABLES/"table1_baseline_unweighted_vs_weighted.csv",index=False,na_rep="Not estimable")
    ess.to_csv(TABLES/"supp_ipcw_weight_distribution_and_ess.csv",index=False)
    attempts.to_csv(TABLES/"weight_selection_attempts.csv",index=False,na_rep="Not applicable")
    frozen={db:freeze(db,data) for db,data in state.items()} if passed else {}
    write_atomic(STATUS,json.dumps({"gate_passed":passed,"databases":d,"input_sha256":protected,
                                    "weight_sha256":frozen},ensure_ascii=False,indent=2,allow_nan=False))
    require(passed,"Reference weight diagnostics failed for one or both databases.")
    require_unblinding_authorization()
    print("PASS. Both weight models passed and the main weights were saved.",flush=True)

if __name__=="__main__":
    run_lock()
