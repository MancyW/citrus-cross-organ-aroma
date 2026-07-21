#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

CFG_LOCO="configs/fig4/base.yaml"
CFG_FITALL="configs/fig4/fitall.yaml"

PANELS=("S1+S4" "S4")

Q=0.05
SPACE="log1p"
MIN_POS=1

echo "=== FINAL PIPELINE ==="
echo "LOCO cfg   : $CFG_LOCO"
echo "FITALL cfg : $CFG_FITALL"
echo "Panels     : ${PANELS[*]}"
echo "Threshold  : q=$Q space=$SPACE min_pos=$MIN_POS"
echo

for PANEL in "${PANELS[@]}"; do
  TAG="${PANEL//+/_}"

  echo "=============================="
  echo "[A] LOCO EVAL panel=$PANEL"
  echo "=============================="
  python -m src.fig4.run.export_pred_vectors --config "$CFG_LOCO" --panel "$PANEL"
  RUN_LOCO="$(ls -dt results/pred_vectors/run_*_"$TAG" | head -1)"
  echo "[LOCO] run_dir=$RUN_LOCO"

  python -m src.fig4.run.qc_alignment --pred_long "$RUN_LOCO/pred_vectors_long.parquet"

  python -m src.fig4.run.ideotype_ranking_v3 \
    --pred_long "$RUN_LOCO/pred_vectors_long.parquet" \
    --ideotype_mode cultivar --anchor MTH \
    --space clr --metric cosine

  python -m src.fig4.run.ideotype_ranking_v3 \
    --pred_long "$RUN_LOCO/pred_vectors_long.parquet" \
    --ideotype_mode cultivar --anchor MTH \
    --space absolute_log1p --metric cosine

  echo
  echo "=============================="
  echo "[B] FITALL TRAIN panel=$PANEL"
  echo "=============================="
  python -m src.fig4.run.export_pred_vectors_fitall --config "$CFG_FITALL" --panel "$PANEL"
  RUN_FITALL="$(ls -dt results/pred_vectors/run_*_"$TAG"_FITALL | head -1)"
  echo "[FITALL] run_dir=$RUN_FITALL"

  python -m src.fig4.run.postprocess_threshold_by_true_quantile \
    --pred_long "$RUN_FITALL/pred_vectors_long.parquet" \
    --quantile "$Q" --space "$SPACE" --min_pos "$MIN_POS" \
    --keep_raw --report

  THR_PATH="$RUN_FITALL/pred_vectors_long.thresh_q${Q}_${SPACE}.parquet"
  if [ ! -f "$THR_PATH" ]; then
    THR_PATH="$(ls "$RUN_FITALL"/pred_vectors_long.thresh_q*_*.parquet | head -1)"
  fi
  echo "[FITALL] thresholded=$THR_PATH"

  python -m src.fig4.run.qc_alignment --pred_long "$THR_PATH"

  python -m src.fig4.run.ideotype_ranking_v3 \
    --pred_long "$THR_PATH" \
    --ideotype_mode cultivar --anchor MTH \
    --space clr --metric cosine

  python -m src.fig4.run.ideotype_ranking_v3 \
    --pred_long "$THR_PATH" \
    --ideotype_mode cultivar --anchor MTH \
    --space absolute_log1p --metric cosine

  echo
done

echo "=== DONE. Key artifacts are in results/pred_vectors/run_*_{panel} and run_*_{panel}_FITALL ==="
