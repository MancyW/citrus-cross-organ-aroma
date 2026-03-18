#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${PROJECT_ROOT}}"
CFG="${2:-$ROOT/configs/base.yaml}"

cd "$ROOT"

python -m src.run.run_final_model --config "$CFG"
echo "[OK] Final model exported. See: results/runs/"
