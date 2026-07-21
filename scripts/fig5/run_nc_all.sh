#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTDIR=""
ANCHOR="MTH"
X_STAGES=()
Y_STAGE=""
RUN_ID="run_20260131_234102_S1_S4"

WEIGHTS_CSV=""
PRED_VECTORS_LONG=""

FOCUS_MODE="weights_topN"
FOCUS_TOPN="30"
SIGN_ALIGN="train_only"

BOOTSTRAP_B="2000"
SEED="0"

RIDGE_ALPHA="1.0"
PLS_COMPONENTS="3"
RF_TREES="500"
SKIP_SKLEARN="0"

DO_SHUFFLE_Y="1"
DO_SHUFFLE_WEIGHTS="1"
SHUFFLE_B="2000"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    --anchor) ANCHOR="$2"; shift 2;;
    --x_stages) shift; while [[ $# -gt 0 && "$1" != --* ]]; do X_STAGES+=("$1"); shift; done;;
    --y_stage) Y_STAGE="$2"; shift 2;;
    --run_id) RUN_ID="$2"; shift 2;;
    --weights_csv) WEIGHTS_CSV="$2"; shift 2;;
    --pred_vectors_long) PRED_VECTORS_LONG="$2"; shift 2;;
    --focus_mode) FOCUS_MODE="$2"; shift 2;;
    --focus_topN) FOCUS_TOPN="$2"; shift 2;;
    --sign_align) SIGN_ALIGN="$2"; shift 2;;
    --bootstrap_B) BOOTSTRAP_B="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --ridge_alpha) RIDGE_ALPHA="$2"; shift 2;;
    --pls_components) PLS_COMPONENTS="$2"; shift 2;;
    --rf_trees) RF_TREES="$2"; shift 2;;
    --skip_sklearn) SKIP_SKLEARN="1"; shift 1;;
    --no_shuffle_y) DO_SHUFFLE_Y="0"; shift 1;;
    --no_shuffle_weights) DO_SHUFFLE_WEIGHTS="0"; shift 1;;
    --shuffle_B) SHUFFLE_B="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

REPO="$(cd "$REPO" && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [[ ${#X_STAGES[@]} -lt 1 ]]; then
  echo "ERROR: --x_stages S1 S4 ..."
  exit 1
fi
if [[ -z "$Y_STAGE" ]]; then
  echo "ERROR: --y_stage S4 ..."
  exit 1
fi

if [[ -z "$WEIGHTS_CSV" ]]; then
  WEIGHTS_CSV="${REPO}/results/pred_vectors/${RUN_ID}/ideotype_v3/voc_weights.weighted_cosine.f1rho.thresh.fpPen0.20.absolute_log1p.rhof1.csv"
fi
if [[ -z "$PRED_VECTORS_LONG" ]]; then
  PRED_VECTORS_LONG="${REPO}/results/pred_vectors/${RUN_ID}/pred_vectors_long.parquet"
fi

if [[ -z "$OUTDIR" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTDIR="${REPO}/results/nc_addons/${TS}"
fi

echo "[NC CLEAN] repo: $REPO"
echo "[NC CLEAN] outdir: $OUTDIR"
echo "[NC CLEAN] anchor=${ANCHOR} | x_stages=${X_STAGES[*]} | y_stage=${Y_STAGE}"
echo "[NC CLEAN] weights_csv=${WEIGHTS_CSV}"
echo "[NC CLEAN] pred_vectors_long=${PRED_VECTORS_LONG}"
mkdir -p "$OUTDIR"

SSOT_LONG_CLEAN="${REPO}/data/ssot/ssot_long.clean.parquet"
SSOT_CS_CLEAN="${REPO}/data/ssot/ssot_cultivar_stage.clean.parquet"

if [[ ! -f "$SSOT_LONG_CLEAN" || ! -f "$SSOT_CS_CLEAN" ]]; then
  echo "[NC CLEAN] clean SSOT not found -> building from data/raw/GCMS_leaf.csv + GCMS_peel.csv (dropping metadata cols)"
  python -m src.fig5_common.00_build_clean_ssot --repo "$REPO"
fi

python -m scripts.fig5.01_interpretability \
  --repo "$REPO" --outdir "$OUTDIR" --anchor "$ANCHOR" \
  --x_stages "${X_STAGES[@]}" --y_stage "$Y_STAGE" \
  --weights_csv "$WEIGHTS_CSV" \
  --bootstrap_B "$BOOTSTRAP_B" --seed "$SEED" --topk 20 --links_B 1000 \
  --ssot_cultivar_stage "$SSOT_CS_CLEAN" \
  --focus_mode "$FOCUS_MODE" --focus_topN "$FOCUS_TOPN" --sign_align "$SIGN_ALIGN"

python -m scripts.fig5.01_dynamics_analysis \
  --repo "$REPO" --outdir "$OUTDIR" --anchor "$ANCHOR" \
  --x_stages "${X_STAGES[@]}" --y_stage "$Y_STAGE" \
  --weights_csv "$WEIGHTS_CSV" --pred_vectors_long "$PRED_VECTORS_LONG" \
  --bootstrap_B "$BOOTSTRAP_B" --seed "$SEED" --ridge_alpha "$RIDGE_ALPHA" \
  --ssot_cultivar_stage "$SSOT_CS_CLEAN" \
  --focus_mode "$FOCUS_MODE" --focus_topN "$FOCUS_TOPN" --sign_align "$SIGN_ALIGN"

python -m scripts.fig5.01_failure_analysis \
  --repo "$REPO" --outdir "$OUTDIR" --anchor "$ANCHOR" \
  --x_stages "${X_STAGES[@]}" --y_stage "$Y_STAGE" \
  --weights_csv "$WEIGHTS_CSV" --pred_vectors_long "$PRED_VECTORS_LONG" \
  --bootstrap_B "$BOOTSTRAP_B" --seed "$SEED" \
  --ssot_cultivar_stage "$SSOT_CS_CLEAN" \
  --focus_mode "$FOCUS_MODE" --focus_topN "$FOCUS_TOPN" --sign_align "$SIGN_ALIGN"

BASELINE_ARGS=()
BASELINE_ARGS+=(--repo "$REPO" --outdir "$OUTDIR" --anchor "$ANCHOR")
BASELINE_ARGS+=(--x_stages "${X_STAGES[@]}" --y_stage "$Y_STAGE")
BASELINE_ARGS+=(--weights_csv "$WEIGHTS_CSV" --pred_vectors_long "$PRED_VECTORS_LONG")
BASELINE_ARGS+=(--bootstrap_B "$BOOTSTRAP_B" --seed "$SEED")
BASELINE_ARGS+=(--ridge_alpha "$RIDGE_ALPHA" --pls_components "$PLS_COMPONENTS" --rf_trees "$RF_TREES")
BASELINE_ARGS+=(--ssot_cultivar_stage "$SSOT_CS_CLEAN")
BASELINE_ARGS+=(--baselines_mode both)
BASELINE_ARGS+=(--focus_mode "$FOCUS_MODE" --focus_topN "$FOCUS_TOPN" --sign_align "$SIGN_ALIGN")
BASELINE_ARGS+=(--shuffle_B "$SHUFFLE_B")
if [[ "$SKIP_SKLEARN" == "1" ]]; then BASELINE_ARGS+=(--skip_sklearn); fi
if [[ "$DO_SHUFFLE_Y" == "1" ]]; then BASELINE_ARGS+=(--do_shuffle_y); fi
if [[ "$DO_SHUFFLE_WEIGHTS" == "1" ]]; then BASELINE_ARGS+=(--do_shuffle_weights); fi

python -m scripts.fig5.01_run_baselines_loco "${BASELINE_ARGS[@]}"

echo "[NC CLEAN] Done. Outputs in: $OUTDIR"
