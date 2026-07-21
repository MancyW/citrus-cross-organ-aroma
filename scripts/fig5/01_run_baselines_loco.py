from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

from src.fig5_common.paths import find_repo_root, ensure_dir

from src.fig5_common import dataio, weights as wio, stats, plotting

def _try_import_sklearn():

    try:

        from sklearn.cross_decomposition import PLSRegression

        from sklearn.ensemble import RandomForestRegressor

        return PLSRegression, RandomForestRegressor

    except Exception:

        return None, None

def main():

    ap = argparse.ArgumentParser(description="Module4: external baselines for ideotype ranking (metric=Spearman on peel_index)")

    ap.add_argument("--repo", default=None)

    ap.add_argument("--outdir", default=None)

    ap.add_argument("--anchor", default="MTH")

    ap.add_argument("--x_stages", nargs="+", required=True)

    ap.add_argument("--y_stage", required=True)

    ap.add_argument("--weights_csv", default=None)

    ap.add_argument("--pred_vectors_long", default=None)

    ap.add_argument("--bootstrap_B", type=int, default=2000)

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--ridge_alpha", type=float, default=1.0)

    ap.add_argument("--pls_components", type=int, default=3)

    ap.add_argument("--rf_trees", type=int, default=500)

    ap.add_argument("--skip_sklearn", action="store_true")

    ap.add_argument("--exclude_cultivars", nargs="*", default=[])

    ap.add_argument("--baselines_mode", default="both", choices=["index","vector","both"])

    ap.add_argument("--do_shuffle_y", action="store_true")

    ap.add_argument("--do_shuffle_weights", action="store_true")

    ap.add_argument("--shuffle_B", type=int, default=2000)

    ap.add_argument("--ssot_cultivar_stage", default=None)

    ap.add_argument("--ssot_long", default=None)

    ap.add_argument("--focus_mode", default="none", choices=["none","weights_topN"])

    ap.add_argument("--focus_topN", type=int, default=30)

    ap.add_argument("--sign_align", default="train_only", choices=["none","train_only"])

    args = ap.parse_args()

    repo = find_repo_root(args.repo)

    outdir = ensure_dir(Path(args.outdir or (repo/"results/nc_addons/tmp_baselines")))

    od = ensure_dir(outdir / "baselines")

    if args.weights_csv is None:

        raise ValueError("--weights_csv is required for baselines.")

    w = wio.load_weights(args.weights_csv)

    focus_vocs = wio.topN_vocs(w, args.focus_topN) if args.focus_mode=="weights_topN" else None

    ssot_cs = dataio.load_ssot_cultivar_stage(args.ssot_cultivar_stage or str(repo/"data/ssot/ssot_cultivar_stage.clean.parquet"))

    peel_true = dataio.subset_matrix(ssot_cs, organ="Peel", stage=args.y_stage, vocs=focus_vocs)

    y_true = dataio.compute_peel_index(peel_true, w, log1p=True)

    X_blocks = []

    for st in args.x_stages:

        Xst = dataio.subset_matrix(ssot_cs, organ="Leaf", stage=st, vocs=focus_vocs).add_prefix(f"{st}::")

        X_blocks.append(Xst)

    X = pd.concat(X_blocks, axis=1).reindex(y_true.index)

    Xv = np.log1p(np.clip(X.to_numpy(float), 0, None))

    keep = [c for c in y_true.index if c not in set(args.exclude_cultivars)]

    Xv = Xv[[y_true.index.get_loc(c) for c in keep], :]

    y = y_true.loc[keep].to_numpy(float)

    cultivars = keep

    yhat_ridge = stats.ridge_fit_predict_loco(Xv, y, alpha=args.ridge_alpha)

    PLSRegression, RandomForestRegressor = (None, None) if args.skip_sklearn else _try_import_sklearn()

    yhat_pls = np.full_like(y, np.nan, float)

    yhat_rf = np.full_like(y, np.nan, float)

    if PLSRegression is not None:

        for i in range(len(y)):

            m = np.ones(len(y), dtype=bool); m[i]=False

            pls = PLSRegression(n_components=min(args.pls_components, Xv.shape[1], m.sum()-1))

            pls.fit(Xv[m], y[m])

            yhat_pls[i] = float(pls.predict(Xv[~m]).ravel()[0])

    if RandomForestRegressor is not None:

        for i in range(len(y)):

            m = np.ones(len(y), dtype=bool); m[i]=False

            rf = RandomForestRegressor(n_estimators=args.rf_trees, random_state=args.seed, n_jobs=-1)

            rf.fit(Xv[m], y[m])

            yhat_rf[i] = float(rf.predict(Xv[~m])[0])

    yhat_ai = np.full_like(y, np.nan, float)

    if args.pred_vectors_long:

        pv = dataio.load_pred_vectors_long(args.pred_vectors_long)

        mat_true = dataio.predvec_to_matrix(pv, "y_true", organ="Peel", stage=args.y_stage).reindex(cultivars)

        mat_pred = dataio.predvec_to_matrix(pv, "y_pred", organ="Peel", stage=args.y_stage).reindex(cultivars)

        yt = dataio.compute_peel_index(mat_true, w, log1p=True).to_numpy(float)

        yp = dataio.compute_peel_index(mat_pred, w, log1p=True).to_numpy(float)

        if args.sign_align == "train_only":

            yp = stats.loco_align_sign(yt, yp)

        yhat_ai = yp

        y = yt

    def summarize(name, yhat):

        rho = stats.spearmanr(y, yhat)

        m, lo, hi, _ = stats.bootstrap_ci_spearman(y, yhat, B=args.bootstrap_B, seed=args.seed+3)

        return {"model": name, "spearman": rho, "spearman_ci025": lo, "spearman_ci975": hi}

    summ = []

    if args.pred_vectors_long:

        summ.append(summarize("AI(pred_vectors_long)", yhat_ai))

    summ.append(summarize("Ridge(leaf->peel_index)", yhat_ridge))

    if PLSRegression is not None:

        summ.append(summarize("PLS(leaf->peel_index)", yhat_pls))

    if RandomForestRegressor is not None:

        summ.append(summarize("RF(leaf->peel_index)", yhat_rf))

    summ_df = pd.DataFrame(summ)

    summ_df.to_csv(od/"baseline_summary.csv", index=False)

    out_pred = pd.DataFrame({"Cultivar": cultivars, "peel_index_true": y})

    if args.pred_vectors_long:

        out_pred["AI_pred"] = yhat_ai

    out_pred["Ridge_pred"] = yhat_ridge

    if PLSRegression is not None:

        out_pred["PLS_pred"] = yhat_pls

    if RandomForestRegressor is not None:

        out_pred["RF_pred"] = yhat_rf

    out_pred.to_csv(od/"baseline_predictions_ai.csv", index=False)

    null_dir = ensure_dir(od/"nulls")

    rng = np.random.default_rng(args.seed)

    if args.do_shuffle_y and args.pred_vectors_long:

        absr = []

        for _ in range(args.shuffle_B):

            ys = rng.permutation(y)

            absr.append(abs(stats.spearmanr(ys, yhat_ai)))

        absr = np.asarray(absr, float)

        pd.DataFrame({"abs_spearman": absr}).to_csv(null_dir/"null_shuffle_y_absrho.csv", index=False)

        plotting.save_hist(absr, str(null_dir/"fig_null_shuffle_y.png"),

                           title="Null: shuffle y (abs Spearman)", xlabel="abs Spearman")

    if args.do_shuffle_weights and args.pred_vectors_long:

        pv = dataio.load_pred_vectors_long(args.pred_vectors_long)

        mat_true = dataio.predvec_to_matrix(pv, "y_true", organ="Peel", stage=args.y_stage).reindex(cultivars)

        mat_pred = dataio.predvec_to_matrix(pv, "y_pred", organ="Peel", stage=args.y_stage).reindex(cultivars)

        wbase = w.dropna().copy().reset_index(drop=True)

        absr = []

        for _ in range(args.shuffle_B):

            wperm = wbase.copy()

            wperm["weight"] = rng.permutation(wperm["weight"].to_numpy(float))

            yt = dataio.compute_peel_index(mat_true, wperm, log1p=True).to_numpy(float)

            yp = dataio.compute_peel_index(mat_pred, wperm, log1p=True).to_numpy(float)

            absr.append(abs(stats.spearmanr(yt, yp)))

        absr = np.asarray(absr, float)

        pd.DataFrame({"abs_spearman": absr}).to_csv(null_dir/"null_shuffle_weights_absrho.csv", index=False)

        plotting.save_hist(absr, str(null_dir/"fig_null_shuffle_weights.png"),

                           title="Null: shuffle weights (abs Spearman)", xlabel="abs Spearman")

    print(f"[OK] wrote baselines outputs to: {od}")

if __name__ == "__main__":

    main()
