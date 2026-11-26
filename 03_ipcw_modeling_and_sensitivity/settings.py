import hashlib
from pathlib import Path

from config import (
    PROCESSED, LAB_CACHE as CACHE, REFERENCE as RESULTS,
    REFERENCE_RUNTIME as RUNTIME, REFERENCE_WEIGHTS as WEIGHTS,
    REFERENCE_TABLES as TABLES, OUTPUTS, MAIN_COHORT as MAIN,
)

SEED = 20261120
MAIN = "W1_6h_main"
EXPECTED = {"mimic":(1820,18608,470), "eicu":(2909,30773,640)}
SMD_LIMIT=0.10
MEAN_RANGE=(0.90,1.10)
ESS_FRACTION=0.50
ESS_ABS=30
def require(ok,message):
    if not ok:
        raise RuntimeError(message)

def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()

def protected_digests():
    paths=sorted(PROCESSED.glob("*.parquet"))
    require(len(paths)==24 and CACHE.is_file(),"Expected 24 cohort and cloning analysis Parquets plus the laboratory cache")
    return {str(p.relative_to(OUTPUTS)):digest(p) for p in [*paths,CACHE]}

def masked(n):
    return "<10" if 0<n<10 else str(int(n))
