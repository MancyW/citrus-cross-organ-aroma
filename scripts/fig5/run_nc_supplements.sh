#!/usr/bin/env bash
set -euo pipefail

MODE="low"
OUTDIR=""
PRED_VECTORS_LONG=""
WEIGHTS_CSV=""
X_STAGES=()
Y_STAGE="S4"
B=20000
SEED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    --pred_vectors_long) PRED_VECTORS_LONG="$2"; shift 2;;
    --weights_csv) WEIGHTS_CSV="$2"; shift 2;;
    --x_stages) shift; while [[ $# -gt 0 && "$1" != --* ]]; do X_STAGES+=("$1"); shift; done;;
    --y_stage) Y_STAGE="$2"; shift 2;;
    --B) B="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    *) echo "[SUPP] Unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ -z "$OUTDIR" ]]; then
  echo "[SUPP] ERROR: --outdir is required" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR_ABS="$(cd "$REPO_ROOT" && python - <<PY
import os
print(os.path.abspath("$OUTDIR"))
PY
)"

echo "[SUPP] repo=$REPO_ROOT"
echo "[SUPP] outdir=$OUTDIR_ABS"
echo "[SUPP] mode=$MODE"

mkdir -p "$OUTDIR_ABS/extras"

python "$REPO_ROOT/nc_scripts/extras/01_make_nc_maintext_snippets.py" \
  --outdir "$OUTDIR_ABS" \
  --x_stages ${X_STAGES[@]+"${X_STAGES[@]}"} \
  --y_stage "$Y_STAGE"

python "$REPO_ROOT/nc_scripts/extras/02_make_nc_tables.py" \
  --outdir "$OUTDIR_ABS"

if [[ "$MODE" == "medium" ]]; then
  if [[ -z "$PRED_VECTORS_LONG" || -z "$WEIGHTS_CSV" ]]; then
    echo "[SUPP] ERROR: medium mode requires --pred_vectors_long and --weights_csv" >&2
    exit 2
  fi

  python "$REPO_ROOT/nc_scripts/extras/10_bootstrap_ai_vs_bestbaseline.py" \
    --outdir "$OUTDIR_ABS" \
    --pred_vectors_long "$PRED_VECTORS_LONG" \
    --weights_csv "$WEIGHTS_CSV" \
    --x_stages ${X_STAGES[@]+"${X_STAGES[@]}"} \
    --y_stage "$Y_STAGE" \
    --B "$B" \
    --seed "$SEED"

  python "$REPO_ROOT/nc_scripts/extras/11_permutation_null.py" \
    --outdir "$OUTDIR_ABS" \
    --weights_csv "$WEIGHTS_CSV" \
    --x_stages ${X_STAGES[@]+"${X_STAGES[@]}"} \
    --y_stage "$Y_STAGE" \
    --B "$B" \
    --seed "$SEED"
fi

echo "[SUPP] Done. Wrote: $OUTDIR_ABS/extras/"
