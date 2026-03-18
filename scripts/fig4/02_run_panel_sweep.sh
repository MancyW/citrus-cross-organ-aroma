#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${PROJECT_ROOT}}"
CFG="${2:-$ROOT/configs/base.yaml}"

cd "$ROOT"

python -m src.run.run_panel_sweep --config "$CFG"
echo "[OK] Panel sweep finished. See: results/runs/"
