#!/usr/bin/env bash
(
set -euo pipefail
THERMOO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
THERMOO_PY="${THERMOO_PYTHON:-/home/tahiti/Malashin_Projects/.venv_a100/bin/python}"
cd "$THERMOO_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
"$THERMOO_PY" -u scripts/install_thermoo.py
"$THERMOO_PY" -u -m thermoo prepare
"$THERMOO_PY" -u -m thermoo verify
"$THERMOO_PY" -u -m thermoo plan
"$THERMOO_PY" -u -m thermoo run "$@" 2>&1 | tee "logs/full_$(date +%Y%m%d_%H%M%S).log"
"$THERMOO_PY" -u -m thermoo analyze --output full --plots
)
