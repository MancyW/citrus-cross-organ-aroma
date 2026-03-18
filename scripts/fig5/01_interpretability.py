from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

from nc_scripts.common.paths import find_repo_root, ensure_dir

from nc_scripts.common import dataio, weights as wio, stats, plotting

def _safe_mean_ci(arr: np.ndarray):

    arr = np.asarray(arr, float)

    m = np.isfinite(arr)

    if m.sum() == 0:

        return np.nan, np.nan, np.nan

    return float(np.nanmean(arr[m])), float(np.nanquantile(arr[m], 0.025)), float(np.nanquantile(arr[m], 0.975))

def main():

    ap = argparse.ArgumentParser(description="Module1: interpretability (leaf biomarkers + leaf-peel links).")

    ap.add_argument("--repo", default=None)

    ap.add_argument("--outdir", required=True)

    ap.add_argument("--anchor", default="MTH")

    ap.add_argument("--x_stages", nargs="+", required=True)

    ap.add_argument("--y_stage", required=True)

    ap.add_argument("--weights_csv", default=None)

    ap.add_argument("--pred_vectors_long", default=None)

    ap.add_argument("--bootstrap_B", type=int, default=2000)

    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--topk", type=int, default=20)

    ap.add_argument("--links_B", type=int, default=1000)

    ap.add_argument("--ssot_cultivar_stage", default=None)

    ap.add_argument("--ssot_long", default=None)

    ap.add_argument("--focus_mode", default="none", choices=["none", "weights_topN"])

    ap.add_argument("--focus_topN", type=int, default=30)

    ap.add_argument("--sign_align", default="train_only", choices=["none", "train_only"])

    args = ap.parse_args()

    repo = find_repo_root(args.repo)

    outdir = ensure_dir(Path(args.outdir))

    od = ensure_dir(outdir / "interpretability")

    ssot_cs = dataio.load_ssot_cultivar_stage(

        args.ssot_cultivar_stage or str(repo / "data" / "ssot" / "ssot_cultivar_stage.clean.parquet")

    )

    if args.weights_csv is None:

        raise ValueError("--weights_csv is required for interpretability.")

    w = wio.load_weights(args.weights_csv)

    focus_vocs = None

    if args.focus_mode == "weights_topN":

        focus_vocs = wio.topN_vocs(w, args.focus_topN)

    peel_true = dataio.subset_matrix(ssot_cs, organ="Peel", stage=args.y_stage, vocs=focus_vocs)

    y_true = dataio.compute_peel_index(peel_true, w, log1p=True)

    leaf_blocks = []

    for st in args.x_stages:

        Xst = dataio.subset_matrix(ssot_cs, organ="Leaf", stage=st, vocs=focus_vocs)

        Xst = Xst.add_prefix(f"{st}::")

        leaf_blocks.append(Xst)

    Xleaf = pd.concat(leaf_blocks, axis=1).loc[y_true.index]

    rows = []

    y = y_true.to_numpy(float)

    for col in Xleaf.columns:

        x = Xleaf[col].to_numpy(float)

        rho = stats.spearmanr(x, y)

        voc = col.split("::", 1)[1] if "::" in col else col

        rows.append({"feature": col, "VOC": voc, "rho": rho, "abs_rho": abs(rho) if np.isfinite(rho) else np.nan})

    biom = pd.DataFrame(rows).sort_values("abs_rho", ascending=False)

    rng = np.random.default_rng(args.seed)

    voc_to_cols = {}

    for c in Xleaf.columns:

        voc = c.split("::", 1)[1] if "::" in c else c

        voc_to_cols.setdefault(voc, []).append(c)

    stab_rows = []

    cultivars = y_true.index.to_list()

    n = len(cultivars)

    for voc, cols in voc_to_cols.items():

        best = biom[biom["VOC"] == voc].sort_values("abs_rho", ascending=False).head(1)

        if best.empty:

            continue

        feat = best["feature"].iloc[0]

        x_full = Xleaf[feat].to_numpy(float)

        rhos = []

        for _ in range(args.bootstrap_B):

            idx = rng.integers(0, n, size=n)

            rhos.append(stats.spearmanr(x_full[idx], y[idx]))

        rhos = np.asarray(rhos, float)

        rho_mean, q025, q975 = _safe_mean_ci(rhos)

        if np.isfinite(rho_mean):

            sign_stab = float(np.nanmean(np.sign(rhos[np.isfinite(rhos)]) == np.sign(rho_mean))) if np.isfinite(rhos).sum() else np.nan

        else:

            sign_stab = np.nan

        stab_rows.append(

            {

                "VOC": voc,

                "feature": feat,

                "rho_mean": rho_mean,

                "rho_q025": q025,

                "rho_q975": q975,

                "sign_stability": sign_stab,

            }

        )

    biom_stab = pd.DataFrame(stab_rows).sort_values("rho_mean", ascending=False)

    if biom_stab.empty:

        links = pd.DataFrame(columns=["leaf_VOC", "peel_VOC", "rho", "abs_rho"])

        links_stab = pd.DataFrame(columns=["leaf_VOC", "peel_VOC", "rho", "rho_q025", "rho_q975"])

    else:

        top_leaf_vocs = biom_stab.sort_values("rho_mean", ascending=False).head(args.topk)["VOC"].tolist()

        w2 = w.dropna().copy()

        w2["absw"] = w2["weight"].abs()

        top_peel_vocs = w2.sort_values("absw", ascending=False).head(args.topk)["VOC"].tolist()

        leaf_voc_vals = {}

        for voc in top_leaf_vocs:

            cols = [c for c in Xleaf.columns if c.endswith(f"::{voc}")]

            if not cols:

                continue

            mat = Xleaf[cols].to_numpy(float)

            val = np.nanmean(mat, axis=1)

            leaf_voc_vals[voc] = np.log1p(np.clip(np.asarray(val, float), 0, None))

        peel_df = dataio.subset_matrix(ssot_cs, organ="Peel", stage=args.y_stage, vocs=top_peel_vocs).reindex(y_true.index)

        peel_cols = peel_df.columns.tolist()

        peel_arr = np.log1p(np.clip(peel_df.to_numpy(float), 0, None))

        peel_mat = pd.DataFrame(peel_arr, index=y_true.index, columns=peel_cols)

        link_rows = []

        for lvoc, x in leaf_voc_vals.items():

            for pvoc in top_peel_vocs:

                yy = peel_mat[pvoc].to_numpy(float)

                rho = stats.spearmanr(x, yy)

                link_rows.append(

                    {"leaf_VOC": lvoc, "peel_VOC": pvoc, "rho": rho, "abs_rho": abs(rho) if np.isfinite(rho) else np.nan}

                )

        links = pd.DataFrame(link_rows).sort_values("abs_rho", ascending=False)

        stab2 = []

        if not links.empty:

            for _, r in links.head(args.topk * args.topk).iterrows():

                lvoc = r["leaf_VOC"]

                pvoc = r["peel_VOC"]

                x = np.asarray(leaf_voc_vals.get(lvoc, np.full(n, np.nan)), float)

                yy = peel_mat[pvoc].to_numpy(float)

                mean_rho, q025, q975, _ = stats.bootstrap_ci_spearman(yy, x, B=args.links_B, seed=args.seed + 1)

                stab2.append({"leaf_VOC": lvoc, "peel_VOC": pvoc, "rho": float(r["rho"]), "rho_q025": q025, "rho_q975": q975})

        links_stab = pd.DataFrame(stab2)

    biom.to_csv(od / "biomarkers_leaf.csv", index=False)

    biom_stab.to_csv(od / "biomarkers_leaf_stability.csv", index=False)

    links.to_csv(od / "leaf_peel_links.csv", index=False)

    links_stab.to_csv(od / "leaf_peel_links_stability.csv", index=False)

    if not biom_stab.empty:

        topplot = biom_stab.copy()

        topplot["abs_rho_mean"] = topplot["rho_mean"].abs()

        plotting.save_barh_top(

            topplot,

            "abs_rho_mean",

            "VOC",

            str(od / "fig_biomarkers_leaf_top20.png"),

            topk=20,

            title=f"Leaf biomarkers vs PeelIndex (Y={args.y_stage})",

        )

    print(f"[OK] wrote interpretability outputs to: {od}")

if __name__ == "__main__":

    main()
