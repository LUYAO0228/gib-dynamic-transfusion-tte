from pathlib import Path

import duckdb
from config import DUCKDB_TEMP, lit, pq, raw, ec, ts

def connect():
    DUCKDB_TEMP.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(config={
        "threads": 5, "memory_limit": "6GB", "temp_directory": str(DUCKDB_TEMP),
        "max_temp_directory_size": "25GB", "preserve_insertion_order": False,
    })
    conn.execute("SET enable_progress_bar=false")
    return conn

c = None

def initialize():
    global c
    if c is None:
        c = connect()

def one(sql):
    return q(sql)[0]

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def save(table, target):
    partial = target.with_suffix(".partial.parquet")
    if partial.exists():
        partial.unlink()
    c.execute(f"COPY {table} TO {lit(partial.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)")
    require(n(table) == one(f"SELECT count(*) FROM read_parquet({lit(partial.as_posix())})")[0],
            f"Parquet row count mismatch: {target.name}")
    partial.replace(target)

def icd(column):
    return f"upper(regexp_replace(trim({column}), '[. ]', '', 'g'))"

def q(sql):
    return c.execute(sql).fetchall()

def build(table, sql):
    c.execute(f"CREATE OR REPLACE TEMP TABLE {table} AS {sql}")

def n(table):
    return q(f"SELECT count(*) FROM {table}")[0][0]

def mask(count):
    return "&lt;10" if 0 < count < 10 else f"{count:,}"
