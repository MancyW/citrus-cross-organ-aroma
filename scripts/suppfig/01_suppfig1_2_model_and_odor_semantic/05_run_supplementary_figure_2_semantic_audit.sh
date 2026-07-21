#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUTDIR="${OUTDIR:-${PROJECT_DIR}/results/supplementary_figure_2_original_space}"

LEAF_DESC_CSV="${LEAF_DESC_CSV:-${PROJECT_DIR}/data/semantic/sample_desc_leaf_relative.csv}"
PEEL_DESC_CSV="${PEEL_DESC_CSV:-${PROJECT_DIR}/data/semantic/sample_desc_peel_relative.csv}"
DESC2D_CSV="${DESC2D_CSV:-${PROJECT_DIR}/data/semantic/sample_desc_2d_relative.csv}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUTDIR}"

echo "[INFO] running original-space semantic transfer audit..."
"${PYTHON_BIN}" "${SCRIPT_DIR}/01_semantic_transfer_from_descriptor_csv.py" \
  --leaf_desc_csv "${LEAF_DESC_CSV}" \
  --peel_desc_csv "${PEEL_DESC_CSV}" \
  --desc2d_csv "${DESC2D_CSV}" \
  --outdir "${OUTDIR}/original_space_transfer" \
  --pair_cols "Cultivar,Stage,Rep" \
  --n_perm 2000 \
  --n_boot 5000 \
  --seed 42

echo "[DONE] original-space semantic transfer audit finished."
