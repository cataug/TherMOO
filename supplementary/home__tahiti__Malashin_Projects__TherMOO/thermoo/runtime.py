"""Use the downloaded source snapshots and the existing scientific environment."""
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("THERMOO_ROOT", Path(__file__).resolve().parents[1])).resolve()
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")
for path in (ROOT / "sources/pymoo", ROOT / "vendor_deps"):
    sys.path.insert(0, str(path))


def code_hash():
    h = hashlib.sha256()
    names = ["__init__.py", "runtime.py", "plan.py", "native.py", "problems.py", "algorithms.py", "control.py", "engine.py"]
    for file in sorted(Path(__file__).parent / name for name in names):
        h.update(file.name.encode())
        h.update(file.read_bytes())
    return h.hexdigest()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4], "little")
