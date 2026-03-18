#!/usr/bin/env bash
set -euo pipefail

cd ${PROJECT_ROOT}

for panel in S1 S2 S3 S4 "S1+S2" "S1+S2+S3+S4"; do
  python -m src.run.export_pred_vectors --config configs/base.yaml --panel "$panel"
done

python -m src.run.panel_robustness_report \
  --panels "S1,S2,S3,S4,S1+S2,S1+S2+S3+S4" \
  --variant BOTH \
  --ideotype_mode cultivar --anchor MTH --top_n 3 \
  --threshold_quantiles "0,0.05" \
  --n_boot 2000 --n_perm 300
