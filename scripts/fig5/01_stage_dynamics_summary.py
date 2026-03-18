from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

from nc_scripts.common.paths import make_outdir, timestamp

from nc_scripts.common.dataio import build_xy, load_long_table, voc_columns, aggregate_cultivar_stage

from nc_scripts.common.weights import parse_weights_csv

from nc_scripts.common.md import df_to_markdown, write_md

from nc_scripts.common.plotting import plot_top_bars, plot_volcano, plot_scatter_with_labels

from nc_scripts.common.stats import (

    bh_fdr,

    exact_signflip_pvalue,

    map_weights_to_matrix_cols,

    safe_spearman,

)

def _strip_prefix(s: str) -> str:

    return str(s).split("::")[-1]

def _peel_index(Y: pd.DataFrame, weights: pd.DataFrame | None) -> pd.Series:

    if weights is None or weights.empty:

        return Y.sum(axis=1).rename("peel_index")

    w, cols = map_weights_to_matrix_cols(weights, Y.columns)

    if len(cols) == 0:

        return Y.sum(axis=1).rename("peel_index")

    return pd.Series(Y[cols].to_numpy(dtype=float) @ w, index=Y.index, name="peel_index")

def _standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    mu = np.nanmean(X_train, axis=0, keepdims=True)

    sd = np.nanstd(X_train - mu, axis=0, ddof=1)

    sd[sd == 0] = 1.0

    return (X_train - mu) / sd, (X_test - mu) / sd

def _ridge_fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float) -> float:

    XtX = X_train.T @ X_train

    p = XtX.shape[0]

    beta = np.linalg.solve(XtX + alpha * np.eye(p), X_train.T @ y_train)

    return float(X_test @ beta)

def loocv_ridge(X: np.ndarray, y: np.ndarray, alpha: float, seed: int = 0) -> np.ndarray:

    X = np.asarray(X, dtype=float)

    y = np.asarray(y, dtype=float)

    n = X.shape[0]

    yhat = np.full(n, np.nan, dtype=float)

    for i in range(n):

        tr = np.array([j for j in range(n) if j != i], dtype=int)

        te = i

        Xtr = X[tr]

        ytr = y[tr]

        Xte = X[[te]]

        col_ok = np.isfinite(Xtr).sum(axis=0) > 0

        Xtr = Xtr[:, col_ok]

        Xte = Xte[:, col_ok]

        mu = np.nanmean(Xtr, axis=0)

        Xtr = np.where(np.isfinite(Xtr), Xtr, mu)

        Xte = np.where(np.isfinite(Xte), Xte, mu)

        Xtr_s, Xte_s = _standardize_train_test(Xtr, Xte)

        yhat[i] = _ridge_fit_predict(Xtr_s, ytr, Xte_s.ravel(), alpha=float(alpha))

    return yhat

def bootstrap_rho_ci(y_true: np.ndarray, y_pred: np.ndarray, B: int, seed: int) -> tuple[float, float]:

    y_true = np.asarray(y_true, dtype=float)

    y_pred = np.asarray(y_pred, dtype=float)

    m = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[m]

    y_pred = y_pred[m]

    n = len(y_true)

    if n < 5:

        return float("nan"), float("nan")

    rng = np.random.default_rng(int(seed))

    rhos = np.empty(int(B), dtype=float)

    for b in range(int(B)):

        idx = rng.integers(0, n, size=n)

        rhos[b] = safe_spearman(y_true[idx], y_pred[idx])

    return float(np.nanquantile(rhos, 0.025)), float(np.nanquantile(rhos, 0.975))

def main() -> None:

    ap = argparse.ArgumentParser(description="Module2 (v1.1): stage dynamics + delta biomarkers")

    ap.add_argument("--repo", default=".")

    ap.add_argument("--outdir", default=None)

    ap.add_argument("--anchor", default=None, help="compat only")

    ap.add_argument("--x_stages", nargs="+", required=True)

    ap.add_argument("--y_stage", required=True)

    ap.add_argument("--weights_csv", default=None)

    ap.add_argument("--bootstrap_B", type=int, default=4000)

    ap.add_argument("--ridge_alpha", type=float, default=1.0)

    ap.add_argument("--topn", type=int, default=30)

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    repo = Path(args.repo).resolve()

    outdir = Path(args.outdir) if args.outdir else repo / "results" / "nc_addons" / timestamp() / "dynamics"

    outdir = make_outdir(outdir)

    if len(args.x_stages) < 2:

        raise ValueError("Module2 needs at least 2 x_stages to define a delta (e.g. S1 S4).")

    early = str(args.x_stages[0])

    late = str(args.x_stages[-1])

    ds = build_xy(

        repo,

        x_stages=list(args.x_stages),

        y_stage=str(args.y_stage),

        organ_x="Leaf",

        organ_y="Peel",

        agg="mean",

        feature_prefix=True,

    )

    X = ds["X"]

    Y = ds["Y"]

    weights = parse_weights_csv(args.weights_csv)

    peel_index = _peel_index(Y, weights)

    vocs = sorted({c.split("::")[-1] for c in X.columns})

    X_early = pd.DataFrame(index=X.index)

    X_late = pd.DataFrame(index=X.index)

    for v in vocs:

        c1 = f"Leaf:{early}::{v}"

        c2 = f"Leaf:{late}::{v}"

        if c1 in X.columns:

            X_early[v] = X[c1]

        if c2 in X.columns:

            X_late[v] = X[c2]

    both = sorted(set(X_early.columns).intersection(set(X_late.columns)))

    X_early = X_early[both]

    X_late = X_late[both]

    delta = X_late - X_early

    rows = []

    for v in both:

        d = delta[v].to_numpy(dtype=float)

        m = np.isfinite(d)

        dv = d[m]

        mu = float(np.nanmean(dv)) if dv.size else float("nan")

        sd = float(np.nanstd(dv, ddof=1)) if dv.size > 1 else float("nan")

        eff = mu / sd if (np.isfinite(mu) and np.isfinite(sd) and sd > 0) else float("nan")

        p = exact_signflip_pvalue(dv)

        rows.append((v, mu, sd, eff, p))

    delta_stats = pd.DataFrame(rows, columns=["voc", "mean_delta", "sd_delta", "effect_d", "p_signflip"])

    delta_stats["q_fdr"] = bh_fdr(delta_stats["p_signflip"].to_numpy(dtype=float))

    delta_stats = delta_stats.sort_values(["q_fdr", "effect_d"], ascending=[True, False]).reset_index(drop=True)

    delta_stats.to_csv(outdir / "delta_voc_stats.csv", index=False)

    yvec = peel_index.to_numpy(dtype=float)

    rows = []

    for v in both:

        rho = safe_spearman(delta[v].to_numpy(dtype=float), yvec)

        rows.append((v, rho))

    delta_assoc = pd.DataFrame(rows, columns=["voc", "rho_delta_vs_peel_index"])

    delta_assoc["abs_rho"] = delta_assoc["rho_delta_vs_peel_index"].abs()

    delta_assoc = delta_assoc.sort_values("abs_rho", ascending=False).reset_index(drop=True)

    delta_assoc.to_csv(outdir / "delta_vs_peel_index.csv", index=False)

    X_static_early = X_early.to_numpy(dtype=float)

    X_static_late = X_late.to_numpy(dtype=float)

    X_delta = delta.to_numpy(dtype=float)

    X_comb = np.hstack([X_static_early, X_static_late, X_delta])

    yhat_early = loocv_ridge(X_static_early, yvec, alpha=float(args.ridge_alpha))

    yhat_late = loocv_ridge(X_static_late, yvec, alpha=float(args.ridge_alpha))

    yhat_delta = loocv_ridge(X_delta, yvec, alpha=float(args.ridge_alpha))

    yhat_comb = loocv_ridge(X_comb, yvec, alpha=float(args.ridge_alpha))

    def _summ(name: str, yhat: np.ndarray) -> dict:

        rho = safe_spearman(yvec, yhat)

        mae = float(np.nanmean(np.abs(yhat - yvec)))

        lo, hi = bootstrap_rho_ci(yvec, yhat, B=int(args.bootstrap_B), seed=int(args.seed) + 17)

        return {"feature_set": name, "spearman_rho": rho, "rho_ci_low": lo, "rho_ci_high": hi, "mae": mae}

    comp = pd.DataFrame([

        _summ(f"leaf_{early}", yhat_early),

        _summ(f"leaf_{late}", yhat_late),

        _summ(f"delta_{late}-{early}", yhat_delta),

        _summ("static+delta", yhat_comb),

    ])

    comp.to_csv(outdir / "model_comparison_peel_index.csv", index=False)

    preds = pd.DataFrame({

        "Cultivar": X.index,

        "peel_index_true": yvec,

        f"pred_leaf_{early}": yhat_early,

        f"pred_leaf_{late}": yhat_late,

        f"pred_delta_{late}-{early}": yhat_delta,

        "pred_static+delta": yhat_comb,

    })

    preds.to_csv(outdir / "peel_index_predictions.csv", index=False)

    topn = int(args.topn)

    plot_top_bars(

        delta_stats.sort_values("effect_d", ascending=False).head(topn).rename(columns={"effect_d": "score"}),

        feature_col="voc",

        score_col="score",

        out_png=outdir / "fig_top_delta_effects.png",

        title=f"Top delta VOCs by effect size (Leaf {late} - {early})",

        max_rows=topn,

    )

    plot_volcano(

        delta_stats,

        x_col="effect_d",

        p_col="p_signflip",

        label_col="voc",

        out_png=outdir / "fig_delta_volcano.png",

        title=f"Delta biomarkers (Leaf {late} - {early})",

        top_labels=12,

    )

    plot_scatter_with_labels(

        x=yvec,

        y=yhat_comb,

        labels=X.index.tolist(),

        out_png=outdir / "fig_peel_index_pred_vs_true.png",

        title="Peel index: LOOCV ridge (static+delta)",

        xlabel="True peel_index",

        ylabel="Predicted peel_index",

        label_top_n=6,

    )

    md = []

    md.append("# Module2: Dynamics (v1.1)\n")

    md.append(f"- leaf stages: `{list(args.x_stages)}` (delta = {late}-{early}) | peel stage: `{args.y_stage}`\n")

    if args.weights_csv:

        md.append(f"- weights_csv: `{args.weights_csv}`\n")

    md.append(f"- ridge_alpha={args.ridge_alpha} | bootstrap_B={args.bootstrap_B} | seed={args.seed}\n")

    md.append("\n## Key outputs\n")

    md.append("- `delta_voc_stats.csv`: VOC-level delta effect + signflip p-value + FDR.\n")

    md.append("- `delta_vs_peel_index.csv`: VOC-level association between delta and peel_index.\n")

    md.append("- `model_comparison_peel_index.csv`: does delta improve LOOCV prediction of peel_index?\n")

    md.append("\n## Top delta biomarkers (by q_fdr then effect)\n")

    md.append(df_to_markdown(delta_stats.head(25)))

    md.append("\n\n## Delta VOCs most associated with peel_index\n")

    md.append(df_to_markdown(delta_assoc.head(25)))

    md.append("\n\n## Static vs delta model comparison\n")

    md.append(df_to_markdown(comp))

    write_md(outdir / "README_dynamics.md", "\n".join(md))

    print(f"[OK] wrote dynamics outputs to: {outdir}")

if __name__ == "__main__":

    main()
