from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

from src.fig5_common.paths import find_repo_root, ensure_dir

from src.fig5_common import dataio, weights as wio, stats, plotting

def main():

    ap = argparse.ArgumentParser(description="Module3: failure analysis + DoA.")

    ap.add_argument("--repo", default=None)

    ap.add_argument("--outdir", required=True)

    ap.add_argument("--anchor", default="MTH")

    ap.add_argument("--x_stages", nargs="+", required=True)

    ap.add_argument("--y_stage", required=True)

    ap.add_argument("--weights_csv", required=True)

    ap.add_argument("--pred_vectors_long", required=True)

    ap.add_argument("--exclude_cultivars", nargs="*", default=[])

    ap.add_argument("--bootstrap_B", type=int, default=2000)

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--ssot_cultivar_stage", default=None)

    ap.add_argument("--ssot_long", default=None)

    ap.add_argument("--focus_mode", default="none", choices=["none","weights_topN"])

    ap.add_argument("--focus_topN", type=int, default=30)

    ap.add_argument("--sign_align", default="train_only", choices=["none","train_only"])

    args = ap.parse_args()

    repo = find_repo_root(args.repo)

    outdir = ensure_dir(Path(args.outdir))

    od = ensure_dir(outdir / "failure")

    ssot_cs = dataio.load_ssot_cultivar_stage(args.ssot_cultivar_stage or str(repo/"data/ssot/ssot_cultivar_stage.clean.parquet"))

    w = wio.load_weights(args.weights_csv)

    focus_vocs = wio.topN_vocs(w, args.focus_topN) if args.focus_mode=="weights_topN" else None

    pv = dataio.load_pred_vectors_long(args.pred_vectors_long)

    mat_true = dataio.predvec_to_matrix(pv, "y_true", organ="Peel", stage=args.y_stage)

    mat_pred = dataio.predvec_to_matrix(pv, "y_pred", organ="Peel", stage=args.y_stage)

    y_true = dataio.compute_peel_index(mat_true, w, log1p=True)

    y_pred = dataio.compute_peel_index(mat_pred, w, log1p=True)

    if args.sign_align == "train_only":

        y_pred = pd.Series(stats.loco_align_sign(y_true.to_numpy(float), y_pred.to_numpy(float)), index=y_true.index)

    keep = [c for c in y_true.index if c not in set(args.exclude_cultivars)]

    y_true = y_true.loc[keep]; y_pred = y_pred.loc[keep]

    mat_true = mat_true.loc[keep]; mat_pred = mat_pred.loc[keep]

    common_vocs = [c for c in mat_true.columns if c in mat_pred.columns]

    T = np.log1p(np.clip(mat_true[common_vocs].to_numpy(float), 0, None))

    P = np.log1p(np.clip(mat_pred[common_vocs].to_numpy(float), 0, None))

    mae_vec = np.nanmean(np.abs(P - T), axis=1)

    abs_err_idx = np.abs(y_pred.to_numpy(float) - y_true.to_numpy(float))

    cultivar_err = pd.DataFrame({

        "Cultivar": y_true.index,

        "vector_mae_log1p": mae_vec,

        "abs_error_peel_index": abs_err_idx,

        "peel_index_true": y_true.to_numpy(float),

        "peel_index_pred": y_pred.to_numpy(float),

    }).sort_values("vector_mae_log1p", ascending=False)

    cultivar_err.to_csv(od/"cultivar_error_summary.csv", index=False)

    st = args.x_stages[-1]

    leaf = dataio.subset_matrix(ssot_cs, organ="Leaf", stage=st, vocs=focus_vocs).reindex(y_true.index)

    X = np.log1p(np.clip(leaf.to_numpy(float), 0, None))

    n = X.shape[0]

    D = np.full((n,n), np.nan, float)

    for i in range(n):

        for j in range(n):

            if i==j: continue

            di = X[i]-X[j]

            D[i,j] = float(np.sqrt(np.nanmean(di*di)))

    doa = np.zeros(n, float)

    for i in range(n):

        d = D[i].copy()

        d[i] = np.nan

        doa[i] = np.nanmin(d)

    cultivar_err2 = cultivar_err.copy()

    cultivar_err2["doa_knn_distance"] = doa

    cultivar_err2.to_csv(od/"cultivar_error_with_doa.csv", index=False)

    rho_abs = stats.spearmanr(cultivar_err2["doa_knn_distance"].to_numpy(float),

                              cultivar_err2["abs_error_peel_index"].to_numpy(float))

    rho_vec = stats.spearmanr(cultivar_err2["doa_knn_distance"].to_numpy(float),

                              cultivar_err2["vector_mae_log1p"].to_numpy(float))

    doa_metrics = pd.DataFrame([{

        "metric": "doa_vs_abs_error_peel_index", "spearman": rho_abs

    },{

        "metric": "doa_vs_vector_mae_log1p", "spearman": rho_vec

    }])

    doa_metrics.to_csv(od/"doa_metrics.csv", index=False)

    infl = cultivar_err2.copy()

    infl["rank_abs_error"] = infl["abs_error_peel_index"].rank(ascending=False, method="average")

    infl["rank_doa"] = infl["doa_knn_distance"].rank(ascending=False, method="average")

    infl["influence_score"] = infl["rank_abs_error"] + infl["rank_doa"]

    infl = infl.sort_values("influence_score", ascending=False)

    infl[["Cultivar","abs_error_peel_index","vector_mae_log1p","doa_knn_distance","influence_score"]].to_csv(

        od/"leave_one_out_influence.csv", index=False

    )

    rho = stats.spearmanr(y_true.to_numpy(float), y_pred.to_numpy(float))

    m, lo, hi, _ = stats.bootstrap_ci_spearman(y_true.to_numpy(float), y_pred.to_numpy(float), B=args.bootstrap_B, seed=args.seed)

    overall = pd.DataFrame([{

        "spearman_peel_index": rho,

        "spearman_ci025": lo,

        "spearman_ci975": hi,

        "mean_vector_mae_log1p": float(np.nanmean(mae_vec)),

        "mean_abs_error_peel_index": float(np.nanmean(abs_err_idx)),

        "n_cultivars": int(len(y_true))

    }])

    overall.to_csv(od/"overall_metrics.csv", index=False)

    plotting.save_hist(np.asarray(doa, float), str(od/"fig_abs_error_vs_doa_knn.png"),

                       title="DoA (KNN distance) distribution", xlabel="doa_knn_distance")

    plotting.save_hist(np.asarray(mae_vec, float), str(od/"fig_vector_mae_vs_doa_knn.png"),

                       title="Vector MAE(log1p) distribution", xlabel="vector_mae_log1p")

    print(f"[OK] wrote failure outputs to: {od}")

if __name__ == "__main__":

    main()
