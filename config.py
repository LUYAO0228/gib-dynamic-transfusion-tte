"""Repository-relative paths and common source readers."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
RAW_MIMIC = DATA / "mimic-iv-3.1"
RAW_EICU = DATA / "eicu-crd-2.0"
PARQUET = DATA / "parquet"

DUCKDB_TEMP = OUTPUTS / "duckdb_temp"
COHORT_STATE = OUTPUTS / "cohort_inputs.duckdb"
PROCESSED = OUTPUTS / "processed"
LAB_CACHE = OUTPUTS / "mimic_candidate_labs.parquet"
REFERENCE = OUTPUTS / "reference"
REFERENCE_RUNTIME = REFERENCE / "runtime"
REFERENCE_WEIGHTS = REFERENCE / "weights"
REFERENCE_TABLES = REFERENCE / "tables"
SENSITIVITY = OUTPUTS / "sensitivity"
SENSITIVITY_INPUTS = SENSITIVITY / "inputs"
SENSITIVITY_RUNTIME = SENSITIVITY / "runtime"
SENSITIVITY_TABLES = SENSITIVITY / "tables"
SCENARIO_WEIGHTS = SENSITIVITY / "weights"
PRIMARY_T0_HOURS = 72
MAIN_COHORT = "W1_6h_main"
DBS = ("mimic", "eicu")


def lit(value):
    return "'" + str(value).replace("'", "''") + "'"


def csv_path(directory, table):
    for suffix in (".csv.gz", ".csv"):
        path = directory / (table + suffix)
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing input {directory.relative_to(ROOT) / table} (.csv.gz or .csv)")


def read_csv(path, **opts):
    options = "".join(f", {key}={lit(value) if isinstance(value, str) else value}" for key, value in opts.items())
    return f"read_csv_auto({lit(path.as_posix())}, header=true{options})"


def raw(group, table, **opts):
    return read_csv(csv_path(RAW_MIMIC / group, table), **opts)


def pq(group, table):
    path = PARQUET / group / (table + ".parquet")
    if path.is_file():
        return f"read_parquet({lit(path.as_posix())})"
    if group in ("MIMIC_hosp", "MIMIC_ICU"):
        return raw(group.split("_")[1].lower(), table)
    raise FileNotFoundError(f"Missing concept export {path.relative_to(ROOT)}")


def ec(table, *, all_varchar=False):
    path = PARQUET / "eICU" / (table + ".parquet")
    if path.is_file() or table == "ventilation_events":
        source = pq("eICU", table)
        return f"(SELECT COLUMNS(*)::VARCHAR FROM {source})" if all_varchar else source
    return read_csv(csv_path(RAW_EICU, table), all_varchar=all_varchar)


def ts(column):
    return (
        f"coalesce(try_strptime(cast({column} AS varchar), "
        "['%d/%m/%Y %H:%M:%S.%f','%d/%m/%Y %H:%M:%S','%d/%m/%Y %H:%M','%d/%m/%Y']), "
        f"try_cast({column} AS timestamp))"
    )
