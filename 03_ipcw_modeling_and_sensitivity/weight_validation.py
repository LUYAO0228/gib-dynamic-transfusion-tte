from settings import *
import json

STATUS = RUNTIME / "blinded_balance_gate.json"

def require_unblinding_authorization():
    """Validate source and weight checksums before reading outcomes."""
    require(STATUS.is_file(),"reference analysis unblinding is locked: no balance decision")
    status=json.loads(STATUS.read_text(encoding="utf-8"))
    require(status.get("gate_passed") is True and all(
        status.get("databases",{}).get(db,{}).get("passes") is True for db in EXPECTED
    ),"reference analysis unblinding is locked: both databases must pass the blinded gate")
    require(status.get("input_sha256")==protected_digests(),
            "reference analysis unblinding is locked: the protected inputs changed")
    require(all((WEIGHTS/f"{db}_locked_main.parquet").is_file() for db in EXPECTED),
            "reference analysis unblinding is locked: frozen weight files are missing")
    require(status.get("weight_sha256")=={db:digest(WEIGHTS/f"{db}_locked_main.parquet") for db in EXPECTED},
            "reference analysis unblinding is locked: frozen weights changed")
    return status


def write_atomic(path, content):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
