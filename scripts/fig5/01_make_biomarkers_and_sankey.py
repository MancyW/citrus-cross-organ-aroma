from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

from nc_scripts.common.paths import make_outdir, timestamp

from nc_scripts.common.dataio import build_xy

from nc_scripts.common.weights import parse_weights_csv

from nc_scripts.common.association import build_leaf_to_peel_links

from nc_scripts.common.plotting import plot_link_heatmap, plot_top_bars, plot_top_bars_ci

from nc_scripts.common.md import df_to_markdown, write_md

from nc_scripts.common.stats import (

    bh_fdr,

    map_weights_to_matrix_cols,

    safe_spearman,

)

def _strip_prefix(feat: str) -> str:

    return str(feat).split("::")[-1]

def _parse_annotation(path: str | None) -> pd.DataFrame | None:

    if not path:

        return None

    p = Path(path)

    if not p.exists():

        return None

    df = pd.read_csv(p)

    if "voc" not in df.columns:

        for c in ["VOC", "name", "feature", "Feature"]:

            if c in df.columns:

                df = df.rename(columns={c: "voc"})

                break

    if "voc" not in df.columns:

        return None

    df = df.copy()

    df["voc"] = df["voc"].astype(str)

    return df

def _peel_index(Y: pd.DataFrame, weights: pd.DataFrame | None) -> pd.Series:

    if weights is None or weights.empty:

        return Y.sum(axis=1).rename("peel_index")

    w, cols = map_weights_to_matrix_cols(weights, Y.columns)

    if len(cols) == 0:

        return Y.sum(axis=1).rename("peel_index")

    return pd.Series(Y[cols].to_numpy(dtype=float) @ w, index=Y.index, name="peel_index")

def _bootstrap_ci_spearman(x: np.ndarray, y: np.ndarray, B: int, seed: int) -> tuple[float, float]:

    x = np.asarray(x, dtype=float)

    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)

    x = x[m]

    y = y[m]

    n = len(x)

    if n < 5:

        return float("nan"), float("nan")

    rng = np.random.default_rng(int(seed))

    rhos = np.empty(int(B), dtype=float)

    for i in range(int(B)):

        idx = rng.integers(0, n, size=n)

        rhos[i] = safe_spearman(x[idx], y[idx])

    lo = float(np.nanquantile(rhos, 0.025))

    hi = float(np.nanquantile(rhos, 0.975))

    return lo, hi

def _perm_pvalue_spearman(x: np.ndarray, y: np.ndarray, B: int, seed: int) -> float:

    x = np.asarray(x, dtype=float)

    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)

    x = x[m]

    y = y[m]

    n = len(x)

    if n < 5:

        return float("nan")

    obs = abs(safe_spearman(x, y))

    rng = np.random.default_rng(int(seed))

    cnt = 0

    for _ in range(int(B)):

        yp = y[rng.permutation(n)]

        if abs(safe_spearman(x, yp)) >= obs - 1e-12:

            cnt += 1

    return float((cnt + 1) / (B + 1))

def main() -> None:

    ap = argparse.ArgumentParser(

        description="Module1 (v1.1): interpretability with CI + stability + leaf→peel linkage"

    )

    ap.add_argument("--repo", default=".", help="AImodel_v2 repo root")

    ap.add_argument("--outdir", default=None)

    ap.add_argument("--anchor", default=None, help="compat only")

    ap.add_argument("--x_stages", nargs="+", required=True)

    ap.add_argument("--y_stage", required=True)

    ap.add_argument("--weights_csv", default=None)

    ap.add_argument("--voc_annotation_csv", default=None)

    ap.add_argument("--topk_leaf", type=int, default=40)

    ap.add_argument("--topk_peel", type=int, default=40)

    ap.add_argument("--linkage", default="spearman", choices=["spearman", "pearson"])

    ap.add_argument("--min_abs_corr", type=float, default=0.25)

    ap.add_argument("--max_links_per_leaf", type=int, default=10)

    ap.add_argument("--bootstrap_B", type=int, default=2000)

    ap.add_argument("--perm_B", type=int, default=4000)

    ap.add_argument("--stability_B", type=int, default=2000)

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    repo = Path(args.repo).resolve()

    outdir = Path(args.outdir) if args.outdir else repo / "results" / "nc_addons" / timestamp() / "interpretability"

    outdir = make_outdir(outdir)

    ds = build_xy(

        repo,

        x_stages=list(args.x_stages),

        y_stage=str(args.y_stage),

        organ_x="Leaf",

        organ_y="Peel",

        agg="mean",

        feature_prefix=True,

    )

    X: pd.DataFrame = ds["X"]

    Y: pd.DataFrame = ds["Y"]

    weights = parse_weights_csv(args.weights_csv)

    ann = _parse_annotation(args.voc_annotation_csv)

    peel_index = _peel_index(Y, weights)

    y_vec = peel_index.to_numpy(dtype=float)

    rows = []

    for feat in X.columns:

        rho = safe_spearman(X[feat].to_numpy(dtype=float), y_vec)

        rows.append((feat, _strip_prefix(feat), rho))

    biom_full = pd.DataFrame(rows, columns=["leaf_feature", "voc", "spearman_rho"])

    biom_full["abs_rho"] = biom_full["spearman_rho"].abs()

    biom_full = biom_full.sort_values("abs_rho", ascending=False).reset_index(drop=True)

    topK = int(args.topk_leaf)

    biom_top = biom_full.head(topK).copy()

    pvals = []

    ci_lo = []

    ci_hi = []

    for i, r in biom_top.iterrows():

        feat = r["leaf_feature"]

        x = X[feat].to_numpy(dtype=float)

        p = _perm_pvalue_spearman(x, y_vec, B=int(args.perm_B), seed=int(args.seed) + 97 * i)

        lo, hi = _bootstrap_ci_spearman(x, y_vec, B=int(args.bootstrap_B), seed=int(args.seed) + 131 * i)

        pvals.append(p)

        ci_lo.append(lo)

        ci_hi.append(hi)

    biom_top["p_perm"] = pvals

    biom_top["q_fdr"] = bh_fdr(biom_top["p_perm"].to_numpy(dtype=float))

    biom_top["ci_low"] = ci_lo

    biom_top["ci_high"] = ci_hi

    rng = np.random.default_rng(int(args.seed))

    feats = X.columns.tolist()

    Xv = X.to_numpy(dtype=float)

    n = len(X)

    counts = np.zeros(len(feats), dtype=int)

    rhos_b = np.empty(len(feats), dtype=float)

    for b in range(int(args.stability_B)):

        idx = rng.integers(0, n, size=n)

        yb = y_vec[idx]

        for j in range(len(feats)):

            rhos_b[j] = safe_spearman(Xv[idx, j], yb)

        top_idx = np.argsort(np.abs(rhos_b))[::-1][:topK]

        counts[top_idx] += 1

    stability = pd.DataFrame(

        {

            "leaf_feature": feats,

            "voc": [ _strip_prefix(f) for f in feats ],

            "freq_in_topk": counts / float(args.stability_B),

        }

    ).sort_values("freq_in_topk", ascending=False).reset_index(drop=True)

    biom_top = biom_top.merge(stability[["leaf_feature", "freq_in_topk"]], on="leaf_feature", how="left")

    if ann is not None:

        biom_top = biom_top.merge(ann, on="voc", how="left")

        stability = stability.merge(ann, on="voc", how="left")

    if weights is not None and not weights.empty:

        w, cols = map_weights_to_matrix_cols(weights, Y.columns)

        if len(cols) > 0:

            biom_peel = pd.DataFrame({"peel_feature": cols, "voc": [ _strip_prefix(c) for c in cols ], "weight": w})

            biom_peel["score"] = biom_peel["weight"].abs()

            biom_peel = biom_peel.sort_values("score", ascending=False).head(int(args.topk_peel)).reset_index(drop=True)

        else:

            peel_var = Y.var(axis=0, numeric_only=True).sort_values(ascending=False)

            biom_peel = pd.DataFrame({"peel_feature": peel_var.index, "voc": [ _strip_prefix(c) for c in peel_var.index ], "score": peel_var.values})

            biom_peel = biom_peel.head(int(args.topk_peel)).reset_index(drop=True)

    else:

        peel_var = Y.var(axis=0, numeric_only=True).sort_values(ascending=False)

        biom_peel = pd.DataFrame({"peel_feature": peel_var.index, "voc": [ _strip_prefix(c) for c in peel_var.index ], "score": peel_var.values})

        biom_peel = biom_peel.head(int(args.topk_peel)).reset_index(drop=True)

    if ann is not None:

        biom_peel = biom_peel.merge(ann, on="voc", how="left")

    links = build_leaf_to_peel_links(

        leaf_mat=X,

        peel_mat=Y,

        leaf_features=biom_top["leaf_feature"].astype(str).tolist(),

        peel_features=biom_peel["peel_feature"].astype(str).tolist(),

        min_abs_corr=float(args.min_abs_corr),

        max_links_per_leaf=int(args.max_links_per_leaf),

        method=str(args.linkage),

    )

    links["leaf_voc"] = links["leaf_feature"].map(_strip_prefix)

    links["peel_voc"] = links["peel_feature"].map(_strip_prefix)

    biom_full.to_csv(outdir / "biomarkers_leaf_full.csv", index=False)

    biom_top.to_csv(outdir / "biomarkers_leaf.csv", index=False)

    stability.to_csv(outdir / "biomarkers_leaf_stability.csv", index=False)

    biom_peel.to_csv(outdir / "biomarkers_peel.csv", index=False)

    links.to_csv(outdir / "leaf_peel_links.csv", index=False)

    plot_top_bars_ci(

        biom_top,

        feature_col="leaf_feature",

        score_col="spearman_rho",

        ci_low_col="ci_low",

        ci_high_col="ci_high",

        out_png=outdir / "fig_leaf_biomarkers_rho_ci.png",

        title="Top leaf biomarkers (Spearman rho to peel_index) with 95% CI",

    )

    plot_top_bars(

        stability.head(topK).rename(columns={"freq_in_topk": "score"}),

        feature_col="leaf_feature",

        score_col="score",

        out_png=outdir / "fig_leaf_biomarkers_stability.png",

        title=f"Leaf biomarker stability (freq in top{topK} over bootstrap)",

        max_rows=topK,

    )

    plot_top_bars(

        biom_peel.rename(columns={"score": "score"}),

        feature_col="peel_feature",

        score_col="score",

        out_png=outdir / "fig_peel_biomarkers.png",

        title="Top peel biomarkers (abs weight or variance)",

        max_rows=int(args.topk_peel),

    )

    plot_link_heatmap(

        links,

        out_png=outdir / "fig_leaf_peel_links_heatmap.png",

        value_col="corr",

        title=f"Leaf→Peel links ({args.linkage})",

    )

    sankey_status = "skipped"

    sankey_note = ""

    try:

        import plotly.graph_objects as go

        if links is not None and not links.empty:

            left = links["leaf_feature"].astype(str).unique().tolist()

            right = links["peel_feature"].astype(str).unique().tolist()

            labels = left + right

            left_idx = {n: i for i, n in enumerate(left)}

            right_idx = {n: i + len(left) for i, n in enumerate(right)}

            src = [left_idx[n] for n in links["leaf_feature"].astype(str).tolist()]

            tgt = [right_idx[n] for n in links["peel_feature"].astype(str).tolist()]

            val = links["abs_corr"].fillna(0.0).tolist()

            fig = go.Figure(

                data=[

                    go.Sankey(

                        node=dict(label=labels, pad=10, thickness=12),

                        link=dict(source=src, target=tgt, value=val, label=links["corr"].round(3).astype(str).tolist()),

                    )

                ]

            )

            fig.update_layout(title_text="Leaf→Peel linkage (abs corr)", font_size=10)

            fig.write_html(outdir / "fig_sankey.html")

            sankey_status = "ok"

        else:

            sankey_status = "empty"

    except Exception as e:

        sankey_status = "unavailable"

        sankey_note = str(e)

    md = []

    md.append("# Module1: Interpretability (v1.1)\n")

    md.append(f"- x_stages: `{list(args.x_stages)}` | y_stage: `{args.y_stage}`\n")

    if args.weights_csv:

        md.append(f"- weights_csv: `{args.weights_csv}`\n")

    if args.voc_annotation_csv:

        md.append(f"- voc_annotation_csv: `{args.voc_annotation_csv}`\n")

    md.append(

        f"- topk_leaf={args.topk_leaf}, topk_peel={args.topk_peel} | linkage={args.linkage} | min_abs_corr={args.min_abs_corr} | max_links_per_leaf={args.max_links_per_leaf}\n"

    )

    md.append(

        f"- bootstrap_B={args.bootstrap_B} (CI), perm_B={args.perm_B} (p-value), stability_B={args.stability_B} | seed={args.seed}\n"

    )

    md.append(f"- sankey: **{sankey_status}**\n")

    if sankey_note:

        md.append(f"  - note: `{sankey_note}`\n")

    md.append("\n## Leaf biomarkers (top)\n")

    md.append(df_to_markdown(biom_top.head(25)))

    md.append("\n\n## Leaf biomarker stability (top)\n")

    md.append(df_to_markdown(stability.head(25)))

    md.append("\n\n## Peel biomarkers (top)\n")

    md.append(df_to_markdown(biom_peel.head(25)))

    md.append("\n\n## Leaf→Peel linkage (top)\n")

    md.append(df_to_markdown(links.head(40)))

    write_md(outdir / "README_interpretability.md", "\n".join(md))

    print(f"[OK] wrote interpretability outputs to: {outdir}")

if __name__ == "__main__":

    main()
