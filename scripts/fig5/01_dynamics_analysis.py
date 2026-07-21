from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

from src.fig5_common.paths import find_repo_root, ensure_dir

from src.fig5_common import dataio, weights as wio, stats

def _concat_leaf(ssot_cs: pd.DataFrame, stages: list[str], vocs: list[str] | None):

    blocks = []

    for st in stages:

        X = dataio.subset_matrix(ssot_cs, organ="Leaf", stage=st, vocs=vocs).add_prefix(f"{st}::")

        blocks.append(X)

    return pd.concat(blocks, axis=1)

def main():

    ap = argparse.ArgumentParser(description="Module2: dynamics (delta features + peel_index prediction comparison).")

    ap.add_argument("--repo", default=None)

    ap.add_argument("--outdir", required=True)

    ap.add_argument("--anchor", default="MTH")

    ap.add_argument("--x_stages", nargs="+", required=True)

    ap.add_argument("--y_stage", required=True)

    ap.add_argument("--weights_csv", required=True)

    ap.add_argument("--pred_vectors_long", default=None)

    ap.add_argument("--bootstrap_B", type=int, default=2000)

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--ridge_alpha", type=float, default=1.0)

    ap.add_argument("--ssot_cultivar_stage", default=None)

    ap.add_argument("--ssot_long", default=None)

    ap.add_argument("--focus_mode", default="none", choices=["none","weights_topN"])

    ap.add_argument("--focus_topN", type=int, default=30)

    ap.add_argument("--sign_align", default="train_only", choices=["none","train_only"])

    args = ap.parse_args()

    repo = find_repo_root(args.repo)

    outdir = ensure_dir(Path(args.outdir))

    od = ensure_dir(outdir / "dynamics")

    ssot_cs = dataio.load_ssot_cultivar_stage(args.ssot_cultivar_stage or str(repo/"data/ssot/ssot_cultivar_stage.clean.parquet"))

    w = wio.load_weights(args.weights_csv)

    focus_vocs = wio.topN_vocs(w, args.focus_topN) if args.focus_mode=="weights_topN" else None

    peel_true = dataio.subset_matrix(ssot_cs, organ="Peel", stage=args.y_stage, vocs=focus_vocs)

    y_true = dataio.compute_peel_index(peel_true, w, log1p=True).sort_index()

    cultivars = y_true.index.tolist()

    y = y_true.to_numpy(float)

    leaf_all = _concat_leaf(ssot_cs, args.x_stages, focus_vocs).loc[cultivars]

    st_low, st_high = None, None

    if len(args.x_stages) >= 2:

        st_low, st_high = args.x_stages[0], args.x_stages[-1]

    X_s_high = dataio.subset_matrix(ssot_cs, organ="Leaf", stage=st_high, vocs=focus_vocs).loc[cultivars]

    X_s_low = dataio.subset_matrix(ssot_cs, organ="Leaf", stage=st_low, vocs=focus_vocs).loc[cultivars]

    A = np.log1p(np.clip(X_s_high.to_numpy(float), 0, None))

    B = np.log1p(np.clip(X_s_low.to_numpy(float), 0, None))

    X_delta = pd.DataFrame(A - B, index=cultivars, columns=X_s_high.columns)

    X_static = pd.DataFrame(A, index=cultivars, columns=[f"{st_high}::{c}" for c in X_s_high.columns])

    X_delta2 = X_delta.add_prefix(f"DELTA({st_high}-{st_low})::")

    X_static_delta = pd.concat([X_static, X_delta2], axis=1)

    yhat_static = stats.ridge_fit_predict_loco(X_static.to_numpy(float), y, alpha=args.ridge_alpha)

    yhat_delta  = stats.ridge_fit_predict_loco(X_delta2.to_numpy(float), y, alpha=args.ridge_alpha)

    yhat_both   = stats.ridge_fit_predict_loco(X_static_delta.to_numpy(float), y, alpha=args.ridge_alpha)

    ai_rho = np.nan

    yhat_ai = np.full_like(y, np.nan, dtype=float)

    if args.pred_vectors_long:

        pv = dataio.load_pred_vectors_long(args.pred_vectors_long)

        mat_true = dataio.predvec_to_matrix(pv, "y_true", organ="Peel", stage=args.y_stage).reindex(cultivars)

        mat_pred = dataio.predvec_to_matrix(pv, "y_pred", organ="Peel", stage=args.y_stage).reindex(cultivars)

        y_ai_true = dataio.compute_peel_index(mat_true, w, log1p=True).to_numpy(float)

        y_ai_pred = dataio.compute_peel_index(mat_pred, w, log1p=True).to_numpy(float)

        if args.sign_align == "train_only":

            y_ai_pred = stats.loco_align_sign(y_ai_true, y_ai_pred)

        yhat_ai = y_ai_pred

        ai_rho = stats.spearmanr(y_ai_true, y_ai_pred)

    comp = []

    for name, yhat in [

        ("LeafStatic(ridge)", yhat_static),

        ("LeafDelta(ridge)",  yhat_delta),

        ("LeafStatic+Delta(ridge)", yhat_both),

    ]:

        rho = stats.spearmanr(y, yhat)

        m, lo, hi, _ = stats.bootstrap_ci_spearman(y, yhat, B=args.bootstrap_B, seed=args.seed)

        comp.append({"model": name, "spearman": rho, "spearman_ci025": lo, "spearman_ci975": hi})

    if args.pred_vectors_long:

        m, lo, hi, _ = stats.bootstrap_ci_spearman(y, yhat_ai, B=args.bootstrap_B, seed=args.seed+7)

        comp.append({"model":"AI(pred_vectors_long)", "spearman": ai_rho, "spearman_ci025": lo, "spearman_ci975": hi})

    comp_df = pd.DataFrame(comp)

    comp_df.to_csv(od/"model_comparison_peel_index.csv", index=False)

    rows = []

    for voc in X_delta.columns:

        rho = stats.spearmanr(X_delta[voc].to_numpy(float), y)

        rows.append({"VOC": voc, "rho_delta_vs_peel_index": rho, "abs_rho": abs(rho) if np.isfinite(rho) else np.nan})

    dv = pd.DataFrame(rows).sort_values("abs_rho", ascending=False)

    dv.to_csv(od/"delta_voc_stats.csv", index=False)

    print(f"[OK] wrote dynamics outputs to: {od}")

if __name__ == "__main__":

    main()
