#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${PROJECT_ROOT}}"
CFG="${2:-$ROOT/configs/base.yaml}"

cd "$ROOT"
python -m src.ssot.build_ssot --config "$CFG"
