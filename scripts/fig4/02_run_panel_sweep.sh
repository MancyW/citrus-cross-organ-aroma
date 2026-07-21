#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${1:-${PROJECT_ROOT:-$DEFAULT_ROOT}}"
CFG="${2:-$ROOT/configs/fig4/base.yaml}"

cd "$ROOT"

python -m src.fig4.run.run_panel_sweep --config "$CFG"
echo "[OK] Panel sweep finished. See: results/runs/"
