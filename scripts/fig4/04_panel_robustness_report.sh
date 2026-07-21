#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for panel in S1 S2 S3 S4 "S1+S2" "S1+S2+S3+S4"; do
  python -m src.fig4.run.export_pred_vectors --config configs/fig4/base.yaml --panel "$panel"
done

python -m src.fig4.run.panel_robustness_report \
  --panels "S1,S2,S3,S4,S1+S2,S1+S2+S3+S4" \
  --variant BOTH \
  --ideotype_mode cultivar --anchor MTH --top_n 3 \
  --threshold_quantiles "0,0.05" \
  --n_boot 2000 --n_perm 300
