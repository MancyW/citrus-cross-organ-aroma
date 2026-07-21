#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

META_CANDIDATES = [
    "SampleID", "Cultivar", "Organ", "Stage", "Batch", "Rep",
    "sample_id", "cultivar", "organ", "stage", "batch", "rep"
]

def normalize_organ(x):
    x = str(x).strip().lower()
    if x.startswith("leaf"):
        return "Leaf"
    if x.startswith("peel"):
        return "Peel"
    return str(x)

def infer_meta_and_desc_cols(df_leaf, df_peel):
    common_cols = [c for c in df_leaf.columns if c in df_peel.columns]
    meta_cols = [c for c in common_cols if c in META_CANDIDATES]

    # descriptor cols = common numeric cols excluding metadata
    desc_cols = []
    for c in common_cols:
        if c in meta_cols:
            continue
        if pd.api.types.is_numeric_dtype(df_leaf[c]) and pd.api.types.is_numeric_dtype(df_peel[c]):
            desc_cols.append(c)

    return meta_cols, desc_cols

def topk_accuracy(sim_matrix, query_keys, candidate_keys, ks=(1, 5, 10)):
    out = {f"top{k}": 0 for k in ks}
    ranks = []

    for i, qk in enumerate(query_keys):
        sims = sim_matrix[i]
        order = np.argsort(-sims)
        ranked_keys = [candidate_keys[j] for j in order]
        try:
            rank = ranked_keys.index(qk) + 1
        except ValueError:
            rank = np.nan
        ranks.append(rank)

        if not np.isnan(rank):
            for k in ks:
                if rank <= k:
                    out[f"top{k}"] += 1

    n = len(query_keys)
    for k in ks:
        out[f"top{k}"] = out[f"top{k}"] / n if n > 0 else np.nan
    ranks = np.asarray(ranks, dtype=float)
    out["mrr"] = np.nanmean(1.0 / ranks)
    out["n"] = n
    return out, ranks

def permutation_null(sim_matrix, query_keys, candidate_keys, ks=(1, 5, 10), n_perm=2000, seed=42):
    rng = np.random.default_rng(seed)
    candidate_keys = np.asarray(candidate_keys, dtype=object)
    rows = []

    for _ in range(n_perm):
        perm_keys = candidate_keys.copy()
        rng.shuffle(perm_keys)
        res, _ = topk_accuracy(sim_matrix, query_keys, list(perm_keys), ks=ks)
        rows.append(res)

    return pd.DataFrame(rows)

def bootstrap_ci(values, func=np.mean, n_boot=5000, seed=42):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(func(values[idx]))
    boots = np.asarray(boots)
    return func(values), np.quantile(boots, 0.025), np.quantile(boots, 0.975)

def bootstrap_metric_ci(ranks, ks=(1, 5, 10), n_boot=5000, seed=42):
    ranks = np.asarray(ranks, dtype=float)
    ranks = ranks[~np.isnan(ranks)]
    rng = np.random.default_rng(seed)
    n = len(ranks)

    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rr = ranks[idx]
        row = {}
        for k in ks:
            row[f"top{k}"] = np.mean(rr <= k)
        row["mrr"] = np.mean(1.0 / rr)
        rows.append(row)

    boot_df = pd.DataFrame(rows)
    out = {}
    for col in boot_df.columns:
        out[f"{col}_lo"] = boot_df[col].quantile(0.025)
        out[f"{col}_hi"] = boot_df[col].quantile(0.975)
    return out

def infer_xy_cols(df):
    candidates = [
        ("UMAP1", "UMAP2"),
        ("x", "y"),
        ("Dim1", "Dim2"),
        ("PC1", "PC2"),
    ]
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            return a, b

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]

    raise ValueError("Cannot infer 2D coordinate columns.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaf_desc_csv", required=True)
    parser.add_argument("--peel_desc_csv", required=True)
    parser.add_argument("--desc2d_csv", default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--pair_cols", default="Cultivar,Stage,Rep")
    parser.add_argument("--n_perm", type=int, default=2000)
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pair_cols = [x.strip() for x in args.pair_cols.split(",") if x.strip()]

    leaf = pd.read_csv(args.leaf_desc_csv)
    peel = pd.read_csv(args.peel_desc_csv)

    if "Organ" not in leaf.columns:
        leaf["Organ"] = "Leaf"
    if "Organ" not in peel.columns:
        peel["Organ"] = "Peel"

    leaf["Organ"] = leaf["Organ"].map(normalize_organ)
    peel["Organ"] = peel["Organ"].map(normalize_organ)

    meta_cols, desc_cols = infer_meta_and_desc_cols(leaf, peel)

    print("[INFO] meta_cols:", meta_cols)
    print("[INFO] n_descriptor_cols:", len(desc_cols))

    # matched sample pairs
    pair_df = leaf.merge(peel, on=pair_cols, suffixes=("_leaf", "_peel"), how="inner")
    if pair_df.shape[0] == 0:
        raise ValueError(f"No matched sample pairs found using pair_cols={pair_cols}")

    leaf_mat = pair_df[[f"{c}_leaf" for c in desc_cols]].to_numpy(dtype=float)
    peel_mat = pair_df[[f"{c}_peel" for c in desc_cols]].to_numpy(dtype=float)

    cos_sim = np.sum(leaf_mat * peel_mat, axis=1) / (
        np.linalg.norm(leaf_mat, axis=1) * np.linalg.norm(peel_mat, axis=1) + 1e-12
    )
    cos_dist = 1.0 - cos_sim
    euc_dist = np.linalg.norm(leaf_mat - peel_mat, axis=1)

    sample_pair_df = pair_df[pair_cols].copy()
    if "SampleID_leaf" in pair_df.columns:
        sample_pair_df["SampleID_leaf"] = pair_df["SampleID_leaf"]
    if "SampleID_peel" in pair_df.columns:
        sample_pair_df["SampleID_peel"] = pair_df["SampleID_peel"]
    sample_pair_df["cosine_similarity_original_space"] = cos_sim
    sample_pair_df["cosine_distance_original_space"] = cos_dist
    sample_pair_df["euclidean_distance_original_space"] = euc_dist
    sample_pair_df.to_csv(outdir / "sample_pair_distances_original_space.csv", index=False)

    # centroid level
    centroid_leaf = leaf.groupby(["Cultivar", "Stage"], as_index=False)[desc_cols].mean()
    centroid_peel = peel.groupby(["Cultivar", "Stage"], as_index=False)[desc_cols].mean()
    centroid_pair = centroid_leaf.merge(centroid_peel, on=["Cultivar", "Stage"], suffixes=("_leaf", "_peel"), how="inner")

    cl = centroid_pair[[f"{c}_leaf" for c in desc_cols]].to_numpy(dtype=float)
    cp = centroid_pair[[f"{c}_peel" for c in desc_cols]].to_numpy(dtype=float)

    cent_cos_sim = np.sum(cl * cp, axis=1) / (
        np.linalg.norm(cl, axis=1) * np.linalg.norm(cp, axis=1) + 1e-12
    )
    cent_cos_dist = 1.0 - cent_cos_sim
    cent_euc_dist = np.linalg.norm(cl - cp, axis=1)

    centroid_dist_df = centroid_pair[["Cultivar", "Stage"]].copy()
    centroid_dist_df["cosine_similarity_original_space"] = cent_cos_sim
    centroid_dist_df["cosine_distance_original_space"] = cent_cos_dist
    centroid_dist_df["euclidean_distance_original_space"] = cent_euc_dist
    centroid_dist_df.to_csv(outdir / "centroid_pair_distances_original_space.csv", index=False)

    # stage summary
    stage_rows = []
    for stage, sub in centroid_dist_df.groupby("Stage"):
        m, lo, hi = bootstrap_ci(sub["cosine_distance_original_space"].to_numpy(), func=np.mean, n_boot=args.n_boot, seed=args.seed)
        stage_rows.append({
            "Stage": stage,
            "n_centroids": len(sub),
            "mean_cosine_distance": m,
            "mean_cosine_distance_ci_lo": lo,
            "mean_cosine_distance_ci_hi": hi,
            "median_cosine_distance": float(np.median(sub["cosine_distance_original_space"].to_numpy())),
            "mean_euclidean_distance": float(np.mean(sub["euclidean_distance_original_space"].to_numpy())),
            "median_euclidean_distance": float(np.median(sub["euclidean_distance_original_space"].to_numpy())),
        })
    stage_summary_df = pd.DataFrame(stage_rows).sort_values("Stage")
    stage_summary_df.to_csv(outdir / "stage_distance_summary.csv", index=False)

    # pairwise tests vs S1
    tests = []
    pivot = centroid_dist_df.pivot(index="Cultivar", columns="Stage", values="cosine_distance_original_space")
    if "S1" in pivot.columns:
        for stage in pivot.columns:
            if stage == "S1":
                continue
            tmp = pivot[["S1", stage]].dropna()
            if len(tmp) >= 3:
                stat, p = wilcoxon(tmp["S1"], tmp[stage], zero_method="wilcox", alternative="two-sided")
            else:
                stat, p = np.nan, np.nan
            tests.append({
                "reference_stage": "S1",
                "compared_stage": stage,
                "n_paired_cultivars": len(tmp),
                "wilcoxon_stat": stat,
                "p_value": p,
            })
    pd.DataFrame(tests).to_csv(outdir / "stage_distance_pairwise_tests.csv", index=False)

    # retrieval: sample level
    leaf_keys = leaf[pair_cols].astype(str).agg("||".join, axis=1).tolist()
    peel_keys = peel[pair_cols].astype(str).agg("||".join, axis=1).tolist()
    sim_sample = cosine_similarity(leaf[desc_cols].to_numpy(dtype=float), peel[desc_cols].to_numpy(dtype=float))
    obs_sample, ranks_sample = topk_accuracy(sim_sample, leaf_keys, peel_keys, ks=(1, 5, 10))
    null_sample = permutation_null(sim_sample, leaf_keys, peel_keys, ks=(1, 5, 10), n_perm=args.n_perm, seed=args.seed)
    boot_sample = bootstrap_metric_ci(ranks_sample, ks=(1, 5, 10), n_boot=args.n_boot, seed=args.seed)

    # retrieval: centroid level
    cent_leaf_keys = centroid_leaf[["Cultivar", "Stage"]].astype(str).agg("||".join, axis=1).tolist()
    cent_peel_keys = centroid_peel[["Cultivar", "Stage"]].astype(str).agg("||".join, axis=1).tolist()
    sim_cent = cosine_similarity(centroid_leaf[desc_cols].to_numpy(dtype=float), centroid_peel[desc_cols].to_numpy(dtype=float))
    obs_cent, ranks_cent = topk_accuracy(sim_cent, cent_leaf_keys, cent_peel_keys, ks=(1, 5, 10))
    null_cent = permutation_null(sim_cent, cent_leaf_keys, cent_peel_keys, ks=(1, 5, 10), n_perm=args.n_perm, seed=args.seed)
    boot_cent = bootstrap_metric_ci(ranks_cent, ks=(1, 5, 10), n_boot=args.n_boot, seed=args.seed)

    rows = []
    for task, obs, null_df, boot in [
        ("sample_pair", obs_sample, null_sample, boot_sample),
        ("cultivar_stage_centroid", obs_cent, null_cent, boot_cent),
    ]:
        row = {"task": task, "n": obs["n"]}
        for metric in ["top1", "top5", "top10", "mrr"]:
            row[metric] = obs[metric]
            row[f"null_{metric}_mean"] = null_df[metric].mean()
            row[f"null_{metric}_lo"] = null_df[metric].quantile(0.025)
            row[f"null_{metric}_hi"] = null_df[metric].quantile(0.975)
            row[f"p_{metric}"] = float((null_df[metric] >= obs[metric]).mean())
        row.update(boot)
        rows.append(row)

    retrieval_df = pd.DataFrame(rows)
    retrieval_df.to_csv(outdir / "retrieval_metrics_original_space.csv", index=False)

    # stagewise centroid retrieval
    stage_rows = []
    for stage in sorted(centroid_leaf["Stage"].unique()):
        lsub = centroid_leaf[centroid_leaf["Stage"] == stage].copy()
        psub = centroid_peel[centroid_peel["Stage"] == stage].copy()
        lk = lsub[["Cultivar", "Stage"]].astype(str).agg("||".join, axis=1).tolist()
        pk = psub[["Cultivar", "Stage"]].astype(str).agg("||".join, axis=1).tolist()
        sm = cosine_similarity(lsub[desc_cols].to_numpy(dtype=float), psub[desc_cols].to_numpy(dtype=float))
        obs, _ = topk_accuracy(sm, lk, pk, ks=(1, 5, 10))
        null_df = permutation_null(sm, lk, pk, ks=(1, 5, 10), n_perm=args.n_perm, seed=args.seed)
        row = {"Stage": stage, "n": obs["n"]}
        for metric in ["top1", "top5", "top10", "mrr"]:
            row[metric] = obs[metric]
            row[f"null_{metric}_mean"] = null_df[metric].mean()
            row[f"null_{metric}_lo"] = null_df[metric].quantile(0.025)
            row[f"null_{metric}_hi"] = null_df[metric].quantile(0.975)
            row[f"p_{metric}"] = float((null_df[metric] >= obs[metric]).mean())
        stage_rows.append(row)
    pd.DataFrame(stage_rows).to_csv(outdir / "retrieval_stagewise_centroid_original_space.csv", index=False)

    # optional comparison with current 2D map
    if args.desc2d_csv is not None:
        d2 = pd.read_csv(args.desc2d_csv)
        if "Organ" in d2.columns:
            d2["Organ"] = d2["Organ"].map(normalize_organ)

        xcol, ycol = infer_xy_cols(d2)

        if "Organ" in d2.columns:
            c2_leaf = d2[d2["Organ"] == "Leaf"].groupby(["Cultivar", "Stage"], as_index=False)[[xcol, ycol]].mean()
            c2_peel = d2[d2["Organ"] == "Peel"].groupby(["Cultivar", "Stage"], as_index=False)[[xcol, ycol]].mean()

            c2 = c2_leaf.merge(c2_peel, on=["Cultivar", "Stage"], suffixes=("_leaf", "_peel"), how="inner")
            c2["umap_euclidean_distance"] = np.sqrt(
                (c2[f"{xcol}_leaf"] - c2[f"{xcol}_peel"]) ** 2 +
                (c2[f"{ycol}_leaf"] - c2[f"{ycol}_peel"]) ** 2
            )

            cmp_df = centroid_dist_df.merge(
                c2[["Cultivar", "Stage", "umap_euclidean_distance"]],
                on=["Cultivar", "Stage"],
                how="inner"
            )

            rho_cos, p_cos = spearmanr(cmp_df["cosine_distance_original_space"], cmp_df["umap_euclidean_distance"])
            rho_euc, p_euc = spearmanr(cmp_df["euclidean_distance_original_space"], cmp_df["umap_euclidean_distance"])

            cmp_df["spearman_rho_cosine_vs_umap"] = rho_cos
            cmp_df["spearman_p_cosine_vs_umap"] = p_cos
            cmp_df["spearman_rho_euclidean_vs_umap"] = rho_euc
            cmp_df["spearman_p_euclidean_vs_umap"] = p_euc
            cmp_df.to_csv(outdir / "distance_comparison_original_vs_umap.csv", index=False)

            plt.figure(figsize=(5, 4))
            plt.scatter(cmp_df["cosine_distance_original_space"], cmp_df["umap_euclidean_distance"], alpha=0.8)
            plt.xlabel("Cosine distance in original descriptor space")
            plt.ylabel("Euclidean distance in 2D semantic map")
            plt.title(f"Spearman rho = {rho_cos:.3f}, p = {p_cos:.3g}")
            plt.tight_layout()
            plt.savefig(outdir / "fig_original_vs_umap_distance_scatter.pdf")
            plt.close()

    # figures
    order = sorted(centroid_dist_df["Stage"].unique())
    data = [centroid_dist_df.loc[centroid_dist_df["Stage"] == s, "cosine_distance_original_space"].to_numpy() for s in order]
    plt.figure(figsize=(5, 4))
    plt.boxplot(data, labels=order, showfliers=True)
    plt.ylabel("Cosine distance in original descriptor space")
    plt.xlabel("Stage")
    plt.tight_layout()
    plt.savefig(outdir / "fig_original_space_distance_by_stage.pdf")
    plt.close()

    r = retrieval_df.set_index("task")
    metrics = ["top1", "top5", "top10"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    for ax, task, title in zip(
        axes,
        ["sample_pair", "cultivar_stage_centroid"],
        ["Sample pair", "Cultivar × Stage centroid"],
    ):
        obs_vals = [r.loc[task, m] for m in metrics]
        null_vals = [r.loc[task, f"null_{m}_mean"] for m in metrics]
        ax.bar(x - width/2, obs_vals, width, label="Observed")
        ax.bar(x + width/2, null_vals, width, label="Null mean")
        ax.set_xticks(x)
        ax.set_xticklabels(["Top-1", "Top-5", "Top-10"])
        ax.set_title(title)
        ax.set_ylabel("Retrieval accuracy")
    axes[0].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outdir / "fig_original_space_retrieval.pdf")
    plt.close()

    with open(outdir / "semantic_original_space_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"n_descriptor_cols = {len(desc_cols)}\n")
        f.write(f"pair_cols = {pair_cols}\n\n")
        f.write("=== retrieval metrics ===\n")
        f.write(retrieval_df.to_string(index=False))
        f.write("\n\n=== stage distance summary ===\n")
        f.write(stage_summary_df.to_string(index=False))
        f.write("\n")

    print("[DONE] original-space analysis completed.")
    print("[INFO] descriptor columns:", len(desc_cols))
    print("[INFO] output dir:", outdir)

if __name__ == "__main__":
    main()