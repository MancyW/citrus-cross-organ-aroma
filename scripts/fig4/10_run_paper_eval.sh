#!/usr/bin/env bash
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
set -euo pipefail

ANCHOR="${ANCHOR:-MTH}"
SPACE="${SPACE:-absolute_log1p}"
WEIGHT_MODE="${WEIGHT_MODE:-rhof1}"
FP_LAM="${FP_LAM:-0.2}"

THRESH_Q="${THRESH_Q:-0.05}"
THRESH_SPACE="${THRESH_SPACE:-log1p}"
THRESH_MIN_POS="${THRESH_MIN_POS:-1}"

OUT_TAG="${OUT_TAG:-weighted_cosine.f1rho.thresh}"

ROOT="${ROOT:-results/pred_vectors}"

pick_latest() {
  local pattern="$1"
  local latest
  latest="$(ls -dt ${ROOT}/${pattern} 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest}" ]]; then
    echo ""
  else
    echo "${latest}"
  fi
}

RUN_LOCO_S1S4="${RUN_LOCO_S1S4:-$(pick_latest 'run_*_S1_S4')}"
RUN_LOCO_S4="${RUN_LOCO_S4:-$(pick_latest 'run_*_S4')}"
RUN_FITALL_S1S4="${RUN_FITALL_S1S4:-$(pick_latest 'run_*_S1_S4_FITALL')}"
RUN_FITALL_S4="${RUN_FITALL_S4:-$(pick_latest 'run_*_S4_FITALL')}"

if [[ "${RUN_LOCO_S1S4}" == *"FITALL"* ]]; then
  RUN_LOCO_S1S4="$(ls -dt ${ROOT}/run_*_S1_S4 2>/dev/null | grep -v FITALL | head -n 1 || true)"
fi
if [[ "${RUN_LOCO_S4}" == *"FITALL"* ]]; then
  RUN_LOCO_S4="$(ls -dt ${ROOT}/run_*_S4 2>/dev/null | grep -v FITALL | head -n 1 || true)"
fi

echo "[PaperEval] Using params:"
echo "  anchor=${ANCHOR} space=${SPACE} weight_mode=${WEIGHT_MODE} fp_penalty_lambda=${FP_LAM}"
echo "  thresh_q=${THRESH_Q} thresh_space=${THRESH_SPACE} thresh_min_pos=${THRESH_MIN_POS}"
echo "  out_tag=${OUT_TAG}"
echo ""
echo "[PaperEval] Run dirs:"
echo "  LOCO  S1+S4 : ${RUN_LOCO_S1S4}"
echo "  LOCO  S4    : ${RUN_LOCO_S4}"
echo "  FITALL S1+S4: ${RUN_FITALL_S1S4}"
echo "  FITALL S4   : ${RUN_FITALL_S4}"
echo ""

run_one () {
  local run_dir="$1"
  if [[ -z "${run_dir}" ]]; then
    echo "[WARN] missing run_dir, skip."
    return 0
  fi

  python scripts/fig4/ideotype_weighted_eval.py \
    --run_dir "${run_dir}" \
    --anchor "${ANCHOR}" \
    --use_thresh \
    --out_tag "${OUT_TAG}" \
    --space "${SPACE}" \
    --weight_mode "${WEIGHT_MODE}" \
    --thresh_q "${THRESH_Q}" \
    --thresh_space "${THRESH_SPACE}" \
    --thresh_min_pos "${THRESH_MIN_POS}" \
    --fp_penalty_lambda "${FP_LAM}"
}

echo "=============================="
echo "[1/4] LOCO S1+S4"
echo "=============================="
run_one "${RUN_LOCO_S1S4}"

echo "=============================="
echo "[2/4] LOCO S4"
echo "=============================="
run_one "${RUN_LOCO_S4}"

echo "=============================="
echo "[3/4] FITALL S1+S4"
echo "=============================="
run_one "${RUN_FITALL_S1S4}"

echo "=============================="
echo "[4/4] FITALL S4"
echo "=============================="
run_one "${RUN_FITALL_S4}"

echo ""
echo "[OK] Paper-default eval finished."
