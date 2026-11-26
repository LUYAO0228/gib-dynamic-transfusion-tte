from settings import *
from weight_validation import require_unblinding_authorization
from trial_data import AuthorizedRead,load_trial
from causal_estimators import estimate,estimate_weights,frozen_weights,primary_def
from concurrent.futures import ProcessPoolExecutor
from sklearn.exceptions import ConvergenceWarning
import hashlib
import pickle
import numpy as np
import argparse

BOOT_CACHE=RUNTIME/"bootstrap"
MAIN_B=500
SENS_B=200
SCENARIOS={
 "mimic":[("S1","W2_6h",None),("S2","W1_4h",None),("S3","W1_6h_ed2h",None),
          ("S4",MAIN,None),("S5a",MAIN,False),("S5b",MAIN,.005),("S5c",MAIN,.025),
          ("S6",MAIN,None),("S7",MAIN,None),("S9-MIMIC",MAIN,None)],
 "eicu":[("S1","W2_6h",None),("S2","W1_4h",None),("S3","W1_6h_ed2h",None),
         ("S4",MAIN,None),("S5a",MAIN,False),("S5b",MAIN,.005),("S5c",MAIN,.025),
         ("S6",MAIN,None),("S8a",MAIN,None),("S8b",MAIN,None),("S9-eICU",MAIN,None)],
}
_TRIAL=None
_EXTENDED=False

def init_worker(t,extended):
    global _TRIAL,_EXTENDED
    _TRIAL=t
    _EXTENDED=extended

def draw(seed):
    t=_TRIAL
    rng=np.random.default_rng(seed)
    weights=np.bincount(rng.integers(t.cluster_frame,size=t.cluster_frame),minlength=t.cluster_frame)[t.clusters]
    try:
        return estimate(t,weights,extended=_EXTENDED)
    except (RuntimeError,ConvergenceWarning,ValueError) as error:
        return {"invalid":type(error).__name__}

def frozen_match(t,reference):
    refit=estimate_weights(t,np.ones(t.n))
    when,evt=primary_def(t)
    relevant=np.isinf(t.art)&(~evt[None,:] | (when[None,:]>t.g))
    require(np.allclose(reference.early,refit.early,rtol=1e-7,atol=1e-9),"Bootstrap grace model disagrees with frozen main specification")
    require(np.allclose(reference.late[relevant],refit.late[relevant],rtol=1e-7,atol=1e-9),
             "Bootstrap terminal model disagrees with frozen weights among contributing clones")

def signature(source):
    names=("trial_data.py","causal_estimators.py","bootstrap_trials.py","model_features.py","weight_selection.py")
    fields=[digest(Path(__file__).with_name(i)) for i in names]
    fields.extend(source.status["weight_sha256"].values())
    return hashlib.sha256(":".join(fields).encode()).hexdigest()

def run_dataset(db,selection,workers):
    source=AuthorizedRead()
    stamp=signature(source)
    BOOT_CACHE.mkdir(exist_ok=True)
    entries=[("Main",MAIN,None)] + SCENARIOS[db]
    for seq,(label,cohort,opt) in enumerate(entries):
        if selection!="all" and label not in selection.split(","):
            continue
        cache=BOOT_CACHE/f"{db}_{label}.pkl"
        if cache.exists():
            with cache.open("rb") as fd:
                prior=pickle.load(fd)
            if prior.get("specification")==stamp:
                print(f"{db} {label}: valid bootstrap checkpoint reused",flush=True)
                continue
        extended=label=="Main"
        t=load_trial(source,db,label,cohort,opt)
        reference=frozen_weights(source.frozen(db),t) if extended else None
        if extended:
            frozen_match(t,reference)
        point=estimate(t,np.ones(t.n),w=reference,extended=extended)
        count=MAIN_B if extended else SENS_B
        root=np.random.SeedSequence([SEED,int(db=="eicu"),seq])
        rng=root.generate_state(count,dtype=np.uint64).tolist()
        print(f"{db} {label}: N={t.n}, launching {count} re-fitted bootstrap draws",flush=True)
        with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(t,extended)) as pool:
            repl=list(pool.map(draw,rng,chunksize=4))
        valid=[r for r in repl if "invalid" not in r]
        require(len(valid)>=.95*count,f"{db} {label}: too many non-estimable bootstrap replications")
        group_sizes={}
        for name,part in t.group.items():
            flags=part if isinstance(part,tuple) else (part,~part)
            group_sizes[name]=[int(z.sum()) for z in flags]
        pack={"specification":stamp,"database":db,"scenario":label,"n":t.n,"grace":t.g,
              "cluster_frame":t.cluster_frame,"clusters_represented":int(len(np.unique(t.clusters))),
              "point":point,"replicates":valid,"attempts":count,"group_sizes":group_sizes}
        tmp=cache.with_suffix(".partial")
        with tmp.open("wb") as fd:
            pickle.dump(pack,fd,protocol=5)
        tmp.replace(cache)
        print(f"{db} {label}: {len(valid)}/{count} usable draws saved",flush=True)
    source.con.close()
    require_unblinding_authorization()

def run_unblinded_causal_analyses():
    require_unblinding_authorization()
    parser=argparse.ArgumentParser()
    parser.add_argument("--database",choices=tuple(EXPECTED),required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--selection",default="all")
    args=parser.parse_args()
    run_dataset(args.database,args.selection,args.workers)

if __name__=="__main__":
    run_unblinded_causal_analyses()
