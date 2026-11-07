"""Install missing helpers into the project, preserving the existing scientific stack."""
from pathlib import Path
import importlib.metadata as metadata
import json
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
target = root / "vendor_deps"
target.mkdir(exist_ok=True)
sys.path.insert(0, str(target))
requirements = [
    ("moocore", "0.3.1", "0.3.2"), ("autograd", "1.4.0", "1.8.0"),
    ("cffi", "1.17.1", "2.0.0"), ("pycparser", "2.21", "2.23"),
    ("platformdirs", "3.0.0", "4.5.0"), ("alive-progress", "3.0.0", "3.3.0"),
    ("about-time", "4.2.1", "4.2.1"), ("graphemeu", "0.7.2", "0.7.2"),
]

def version_tuple(value):
    parts = [int(v) for v in re.findall(r"\d+", value)[:3]]
    return tuple(parts + [0] * (3-len(parts)))

core = {}
for name in ("numpy", "scipy"):
    try:
        core[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        raise RuntimeError(f"{name} is missing in {sys.executable}; select the existing .venv_a100 interpreter")
if version_tuple(core["numpy"]) < (1,24,0) or version_tuple(core["scipy"]) < (1,10,0):
    raise RuntimeError(f"Existing NumPy/SciPy are too old: {core}")
pending = []
for name, minimum, pinned in requirements:
    try:
        installed = metadata.version(name)
    except metadata.PackageNotFoundError:
        installed = "0"
    if version_tuple(installed) < version_tuple(minimum):
        pending.append(f"{name}=={pinned}")
    else:
        print(f"[KEEP] {name}=={installed}", flush=True)
if pending:
    print("[LOCAL DEPENDENCIES] " + ", ".join(pending), flush=True)
    subprocess.run([sys.executable, "-m", "pip", "--disable-pip-version-check", "install", "--no-deps", "--upgrade", "--target", str(target), *pending], check=True)
for name, old in core.items():
    if metadata.version(name) != old:
        raise RuntimeError(f"Unexpected change to {name}")
for path in ("sources/pymoo/pymoo", "sources/ZCAT/src", "sources/jMetalPy", "configs/reference_fronts_main.json"):
    if not (root/path).exists():
        raise FileNotFoundError(f"Missing downloaded input: {root/path}")
(root/"reports").mkdir(exist_ok=True)
(root/"reports/runtime_install.json").write_text(json.dumps(dict(python=sys.executable, existing_core=core, project_dependencies=pending), indent=2))
print(f"[READY] Existing core preserved: {core}", flush=True)
