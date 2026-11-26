import hashlib
import json
import pickle

import duckdb
import numpy as np
import pandas as pd

from config import (
    REFERENCE, SENSITIVITY_TABLES as TABLES, SENSITIVITY_INPUTS, SENSITIVITY_RUNTIME as WORK,
    SCENARIO_WEIGHTS, DUCKDB_TEMP, PROCESSED, MAIN_COHORT as COHORT, DBS,
)

def connect():
    return duckdb.connect(config={
        "threads": 3, "memory_limit": "6GB", "preserve_insertion_order": False,
        "temp_directory": str(DUCKDB_TEMP),
    })

def processed(db, kind):
    return PROCESSED / f"{db}_{kind}_{COHORT}.parquet"

def mask(n):
    return "<10" if 0 < n < 10 else int(n)

def prop(n,d):
    return np.nan if d==0 or 0<n<10 or 0<d-n<10 else 100*n/d

def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def dump(obj, path):
    part=path.with_suffix(".part")
    with part.open("wb") as f:
        pickle.dump(obj,f,protocol=5)
    part.replace(path)

def load(path):
    with path.open("rb") as f:
        return pickle.load(f)

def tail(samples, null):
    z = np.asarray(samples,float)
    z = z[np.isfinite(z)]
    n = len(z)
    if n == 0:
        return {"boot_n":0,"lo":np.nan,"hi":np.nan,"raw_tail_crossings":"","p_empirical_plus1":np.nan,"p_formatted":""}
    le,ge = int(sum(z<=null)),int(sum(z>=null))
    p=min(1.,2*(min(le,ge)+1)/(n+1))
    formatted = ("<0.001" if n==1000 else "<0.002") if min(le,ge)==0 and n in (500,1000) else f"{p:.4f}"
    lo,hi=np.quantile(z,[.025,.975])
    return {"boot_n":n,"lo":lo,"hi":hi,"raw_tail_crossings":f"{min(le,ge)} / {n}",
            "k_le_null":le,"k_ge_null":ge,"p_empirical_plus1":p,"p_formatted":formatted}
