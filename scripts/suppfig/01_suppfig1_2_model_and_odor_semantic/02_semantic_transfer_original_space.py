#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, spearmanr
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# utilities
# ---------------------------

def normalize_rows(x):
    x = np.asarray(x, dtype=float)
    row_sums = x.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return x / row_sums

def bootstrap_ci(values, func=np.mean, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boots = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(func(values[idx]))
    boots = np.asarray(boots)
    return func(values), np.quantile(boots, 0.025), np.quantile(boots, 0.975)

def make_weights(x, scheme="relative", topk=None):
    x = np.asarray(x, dtype=float)
    x = np.where(np.isnan(x), 0.0, x)
    x = np.clip(x, 0.0, None)

    if scheme == "relative":
        z = x.copy()

    elif scheme == "log1p":
        z = np.log1p(x)

    elif scheme == "sqrt":
        z = np.sqrt(x)

    elif scheme == "relative_top20":
        z = x.copy()
        k = 20 if topk is None else topk
        if z.shape[1] > k:
            order = np.argsort(-z, axis=1)
            mask = np.zeros_like(z, dtype=bool)
            rows = np.arange(z.shape[0])[:, None]
            mask[rows, order[:, :k]] = True
            z[~mask] = 0.0

    elif scheme == "relative_top50":
        z = x.copy()
        k = 50 if topk is None else topk
        if z.shape[1] > k:
            order = np.argsort(-z, axis=1)
            mask = np.zeros_like(z, dtype=bool)
            rows = np.arange(z.shape[0])[:, None]
            mask[rows, order[:, :k]] = True
            z[~mask] = 0.0
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")

    return normalize_rows(z)

def topk_accuracy(sim_matrix, query_keys, candidate_keys, ks=(1, 5, 10)):
    out = {f"top{k}": 0 for k in ks}
    ranks = []

    # for each query, true match = exactly same key in candidate pool
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
    mrr = np.nanmean(1.0 / ranks)
    out["mrr"] = mrr
    out["n"] = n
    return out, ranks

def permutation_null(sim_matrix, query_keys, candidate_keys, ks=(1, 5, 10), n_perm=2000, seed=42):
    rng = np.random.default_rng(seed)
    candidate_keys = np.asarray(candidate_keys, dtype=object)

    null_rows = []
    for _ in range(n_perm):
        perm_keys = candidate_keys.copy()
        rng.shuffle(perm_keys)
        res, _ = topk_accuracy(sim_matrix, query_keys, list(perm_keys), ks=ks)
        null_rows.append(res)

    return pd.DataFrame(null_rows)

def bootstrap_metric_ci(ranks, ks=(1, 5, 10), n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    ranks = np.asarray(ranks, dtype=float)
    valid = ~np.isnan(ranks)
    ranks = ranks[valid]

    if len(ranks) == 0:
        return {}

    rows = []
    n = len(ranks)
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

def load_descriptor_matrix(path):
    df = pd.read_csv(path)
    if "VOC" not in df.columns:
        raise ValueError("Descriptor matrix must contain a 'VOC' column.")
    desc_cols = [c for c in df.columns if c != "VOC"]
    mat = df.set_index("VOC")[desc_cols]
    return mat, desc_cols

def harmonize_organ_label(x):
    x = str(x).strip().lower()
    if x.startswith("leaf"):
        return "Leaf"
    if x.startswith("peel"):
        return "Peel"
    return str(x)

# ---------------------------
# main analysis
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Original-space odor-descriptor transfer and retrieval analysis."
    )
    parser.add_argument("--sample_matrix", required=True, type=str,
                        help="Wide sample x VOC CSV with metadata columns.")
    parser.add_argument("--descriptor_matrix", required=True, type=str,
                        help="VOC x descriptor probability CSV; first column must be VOC.")
    parser.add_argument("--outdir", required=True, type=str)
    parser.add_argument("--meta_cols", type=str,
                        default="SampleID,Cultivar,Organ,Stage,Rep",
                        help="Comma-separated metadata columns present in sample_matrix.")
    parser.add_argument("--pair_cols", type=str,
                        default="Cultivar,Stage,Rep",
                        help="Columns defining matched leaf-peel sample pairs.")
    parser.add_argument("--weighting", type=str, default="relative",
                        choices=["relative", "log1p", "sqrt", "relative_top20", "relative_top50"])
    parser.add_argument("--umap_centroid_csv", type=str, default=None,
                        help="Optional centroid UMAP CSV with Cultivar,Stage,Organ,UMAP1,UMAP2.")
    parser.add_argument("--n_perm", type=int, default=2000)
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta_cols = [x.strip() for x in args.meta_cols.split(",") if x.strip()]
    pair_cols = [x.strip() for x in args.pair_cols.split(",") if x.strip()]

    # 1) load data
    sample_df = pd.read_csv(args.sample_matrix)
    desc_df, descriptor_cols = load_descriptor_matrix(args.descriptor_matrix)

    # harmonize organ labels
    if "Organ" in sample_df.columns:
        sample_df["Organ"] = sample_df["Organ"].map(harmonize_organ_label)

    missing_meta = [c for c in meta_cols if c not in sample_df.columns]
    if missing_meta:
        raise ValueError(f"Missing metadata columns in sample_matrix: {missing_meta}")

    # intersection VOCs
    voc_cols = [c for c in sample_df.columns if c in desc_df.index]
    if len(voc_cols) == 0:
        raise ValueError("No overlapping VOC columns between sample_matrix and descriptor_matrix.")

    X = sample_df[voc_cols].fillna(0).to_numpy(dtype=float)
    X = np.clip(X, 0.0, None)

    desc_mat = desc_df.loc[voc_cols, descriptor_cols].to_numpy(dtype=float)

    # 2) build sample odor profiles in original descriptor space
    W = make_weights(X, scheme=args.weighting)
    odor_profiles = W @ desc_mat  # n_samples x n_descriptors

    prof_df = sample_df[meta_cols].copy()
    for i, d in enumerate(descriptor_cols):
        prof_df[d] = odor_profiles[:, i]

    prof_path = outdir / f"sample_profiles_{args.weighting}.csv"
    prof_df.to_csv(prof_path, index=False)

    # 3) matched sample-level pairs
    leaf = prof_df[prof_df["Organ"] == "Leaf"].copy()
    peel = prof_df[prof_df["Organ"] == "Peel"].copy()

    leaf_desc = leaf[descriptor_cols].copy()
    peel_desc = peel[descriptor_cols].copy()

    leaf_key = leaf[pair_cols].astype(str).agg("||".join, axis=1)
    peel_key = peel[pair_cols].astype(str).agg("||".join, axis=1)

    leaf = leaf.assign(_pairkey=leaf_key.values)
    peel = peel.assign(_pairkey=peel_key.values)

    pair_df = leaf.merge(
        peel,
        on=pair_cols,
        suffixes=("_leaf", "_peel"),
        how="inner",
    )

    if pair_df.shape[0] == 0:
        raise ValueError("No matched leaf-peel sample pairs found using pair_cols.")

    leaf_pair_mat = pair_df[[f"{d}_leaf" for d in descriptor_cols]].to_numpy(dtype=float)
    peel_pair_mat = pair_df[[f"{d}_peel" for d in descriptor_cols]].to_numpy(dtype=float)

    # cosine distance / euclidean distance in original descriptor space
    cos_sim_pair = np.sum(leaf_pair_mat * peel_pair_mat, axis=1) / (
        np.linalg.norm(leaf_pair_mat, axis=1) * np.linalg.norm(peel_pair_mat, axis=1) + 1e-12
    )
    cos_dist_pair = 1.0 - cos_sim_pair
    euc_dist_pair = np.linalg.norm(leaf_pair_mat - peel_pair_mat, axis=1)

    pair_dist_df = pair_df[pair_cols + ["SampleID_leaf", "SampleID_peel"]].copy()
    pair_dist_df["cosine_similarity_original_space"] = cos_sim_pair
    pair_dist_df["cosine_distance_original_space"] = cos_dist_pair
    pair_dist_df["euclidean_distance_original_space"] = euc_dist_pair
    pair_dist_path = outdir / "sample_pair_distances_original_space.csv"
    pair_dist_df.to_csv(pair_dist_path, index=False)

    # 4) centroid-level profiles
    centroid_df = (
        prof_df.groupby(["Cultivar", "Stage", "Organ"], as_index=False)[descriptor_cols]
        .mean()
    )
    centroid_path = outdir / f"centroid_profiles_{args.weighting}.csv"
    centroid_df.to_csv(centroid_path, index=False)

    leaf_c = centroid_df[centroid_df["Organ"] == "Leaf"].copy()
    peel_c = centroid_df[centroid_df["Organ"] == "Peel"].copy()
    cent_pair = leaf_c.merge(peel_c, on=["Cultivar", "Stage"], suffixes=("_leaf", "_peel"), how="inner")

    leaf_cent_mat = cent_pair[[f"{d}_leaf" for d in descriptor_cols]].to_numpy(dtype=float)
    peel_cent_mat = cent_pair[[f"{d}_peel" for d in descriptor_cols]].to_numpy(dtype=float)

    cos_sim_cent = np.sum(leaf_cent_mat * peel_cent_mat, axis=1) / (
        np.linalg.norm(leaf_cent_mat, axis=1) * np.linalg.norm(peel_cent_mat, axis=1) + 1e-12
    )
    cos_dist_cent = 1.0 - cos_sim_cent
    euc_dist_cent = np.linalg.norm(leaf_cent_mat - peel_cent_mat, axis=1)

    cent_dist_df = cent_pair[["Cultivar", "Stage"]].copy()
    cent_dist_df["cosine_similarity_original_space"] = cos_sim_cent
    cent_dist_df["cosine_distance_original_space"] = cos_dist_cent
    cent_dist_df["euclidean_distance_original_space"] = euc_dist_cent
    cent_dist_path = outdir / "centroid_pair_distances_original_space.csv"
    cent_dist_df.to_csv(cent_dist_path, index=False)

    # 5) stage summary (centroid-level as primary)
    stage_rows = []
    for stage, sub in cent_dist_df.groupby("Stage"):
        mean_cos, cos_lo, cos_hi = bootstrap_ci(
            sub["cosine_distance_original_space"].to_numpy(),
            func=np.mean, n_boot=args.n_boot, seed=args.seed
        )
        med_cos, med_cos_lo, med_cos_hi = bootstrap_ci(
            sub["cosine_distance_original_space"].to_numpy(),
            func=np.median, n_boot=args.n_boot, seed=args.seed
        )
        mean_euc, euc_lo, euc_hi = bootstrap_ci(
            sub["euclidean_distance_original_space"].to_numpy(),
            func=np.mean, n_boot=args.n_boot, seed=args.seed
        )
        med_euc, med_euc_lo, med_euc_hi = bootstrap_ci(
            sub["euclidean_distance_original_space"].to_numpy(),
            func=np.median, n_boot=args.n_boot, seed=args.seed
        )
        stage_rows.append({
            "Stage": stage,
            "n_centroids": len(sub),
            "mean_cosine_distance": mean_cos,
            "mean_cosine_distance_ci_lo": cos_lo,
            "mean_cosine_distance_ci_hi": cos_hi,
            "median_cosine_distance": med_cos,
            "median_cosine_distance_ci_lo": med_cos_lo,
            "median_cosine_distance_ci_hi": med_cos_hi,
            "mean_euclidean_distance": mean_euc,
            "mean_euclidean_distance_ci_lo": euc_lo,
            "mean_euclidean_distance_ci_hi": euc_hi,
            "median_euclidean_distance": med_euc,
            "median_euclidean_distance_ci_lo": med_euc_lo,
            "median_euclidean_distance_ci_hi": med_euc_hi,
        })
    stage_summary_df = pd.DataFrame(stage_rows).sort_values("Stage")
    stage_summary_path = outdir / "stage_distance_summary.csv"
    stage_summary_df.to_csv(stage_summary_path, index=False)

    # pairwise stage tests (paired by cultivar)
    stage_test_rows = []
    pivot_cos = cent_dist_df.pivot(index="Cultivar", columns="Stage", values="cosine_distance_original_space")
    stages = list(stage_summary_df["Stage"])
    if "S1" in stages:
        base = "S1"
        for s in stages:
            if s == base:
                continue
            tmp = pivot_cos[[base, s]].dropna()
            if len(tmp) >= 3:
                stat, p = wilcoxon(tmp[base], tmp[s], zero_method="wilcox", alternative="two-sided")
            else:
                stat, p = np.nan, np.nan
            stage_test_rows.append({
                "reference_stage": base,
                "compared_stage": s,
                "n_paired_cultivars": len(tmp),
                "wilcoxon_stat": stat,
                "p_value": p,
            })
    stage_tests_df = pd.DataFrame(stage_test_rows)
    stage_tests_path = outdir / "stage_distance_pairwise_tests.csv"
    stage_tests_df.to_csv(stage_tests_path, index=False)

    # 6) retrieval in original descriptor space
    # sample-level
    leaf_sample_keys = leaf[pair_cols].astype(str).agg("||".join, axis=1).tolist()
    peel_sample_keys = peel[pair_cols].astype(str).agg("||".join, axis=1).tolist()

    leaf_sample_mat = leaf[descriptor_cols].to_numpy(dtype=float)
    peel_sample_mat = peel[descriptor_cols].to_numpy(dtype=float)

    sim_sample = cosine_similarity(leaf_sample_mat, peel_sample_mat)
    sample_obs, sample_ranks = topk_accuracy(sim_sample, leaf_sample_keys, peel_sample_keys, ks=(1, 5, 10))
    sample_null = permutation_null(
        sim_sample, leaf_sample_keys, peel_sample_keys,
        ks=(1, 5, 10), n_perm=args.n_perm, seed=args.seed
    )
    sample_boot = bootstrap_metric_ci(sample_ranks, ks=(1, 5, 10), n_boot=args.n_boot, seed=args.seed)

    # centroid-level
    leaf_cent_keys = leaf_c[["Cultivar", "Stage"]].astype(str).agg("||".join, axis=1).tolist()
    peel_cent_keys = peel_c[["Cultivar", "Stage"]].astype(str).agg("||".join, axis=1).tolist()

    leaf_cent_full = leaf_c[descriptor_cols].to_numpy(dtype=float)
    peel_cent_full = peel_c[descriptor_cols].to_numpy(dtype=float)

    sim_cent = cosine_similarity(leaf_cent_full, peel_cent_full)
    cent_obs, cent_ranks = topk_accuracy(sim_cent, leaf_cent_keys, peel_cent_keys, ks=(1, 5, 10))
    cent_null = permutation_null(
        sim_cent, leaf_cent_keys, peel_cent_keys,
        ks=(1, 5, 10), n_perm=args.n_perm, seed=args.seed
    )
    cent_boot = bootstrap_metric_ci(cent_ranks, ks=(1, 5, 10), n_boot=args.n_boot, seed=args.seed)

    # p-values (one-sided: null >= obs)
    retrieval_rows = []
    for task_name, obs, null_df, boot in [
        ("sample_pair", sample_obs, sample_null, sample_boot),
        ("cultivar_stage_centroid", cent_obs, cent_null, cent_boot),
    ]:
        row = {"task": task_name, "n": obs["n"]}
        for metric in ["top1", "top5", "top10", "mrr"]:
            row[metric] = obs[metric]
            row[f"null_{metric}_mean"] = null_df[metric].mean()
            row[f"null_{metric}_lo"] = null_df[metric].quantile(0.025)
            row[f"null_{metric}_hi"] = null_df[metric].quantile(0.975)
            row[f"p_{metric}"] = float((null_df[metric] >= obs[metric]).mean())
            if metric in boot:
                pass
        row.update(boot)
        retrieval_rows.append(row)

    retrieval_df = pd.DataFrame(retrieval_rows)
    retrieval_path = outdir / "retrieval_metrics_original_space.csv"
    retrieval_df.to_csv(retrieval_path, index=False)

    # stagewise centroid retrieval
    stagewise_rows = []
    for stage in sorted(leaf_c["Stage"].unique()):
        ls = leaf_c[leaf_c["Stage"] == stage].copy()
        ps = peel_c[peel_c["Stage"] == stage].copy()
        lk = ls[["Cultivar", "Stage"]].astype(str).agg("||".join, axis=1).tolist()
        pk = ps[["Cultivar", "Stage"]].astype(str).agg("||".join, axis=1).tolist()
        sm = cosine_similarity(ls[descriptor_cols].to_numpy(dtype=float),
                               ps[descriptor_cols].to_numpy(dtype=float))
        obs, ranks = topk_accuracy(sm, lk, pk, ks=(1, 5, 10))
        null_df = permutation_null(sm, lk, pk, ks=(1, 5, 10), n_perm=args.n_perm, seed=args.seed)
        row = {"Stage": stage, "n": obs["n"]}
        for metric in ["top1", "top5", "top10", "mrr"]:
            row[metric] = obs[metric]
            row[f"null_{metric}_mean"] = null_df[metric].mean()
            row[f"null_{metric}_lo"] = null_df[metric].quantile(0.025)
            row[f"null_{metric}_hi"] = null_df[metric].quantile(0.975)
            row[f"p_{metric}"] = float((null_df[metric] >= obs[metric]).mean())
        stagewise_rows.append(row)
    stagewise_df = pd.DataFrame(stagewise_rows)
    stagewise_path = outdir / "retrieval_stagewise_centroid_original_space.csv"
    stagewise_df.to_csv(stagewise_path, index=False)

    # 7) optional comparison with current UMAP centroid distances
    if args.umap_centroid_csv is not None:
        umap_df = pd.read_csv(args.umap_centroid_csv)
        if "Organ" in umap_df.columns:
            umap_df["Organ"] = umap_df["Organ"].map(harmonize_organ_label)

        need_cols = {"Cultivar", "Stage", "Organ", "UMAP1", "UMAP2"}
        if not need_cols.issubset(set(umap_df.columns)):
            raise ValueError(f"UMAP centroid CSV must contain columns: {sorted(need_cols)}")

        ul = umap_df[umap_df["Organ"] == "Leaf"].copy()
        up = umap_df[umap_df["Organ"] == "Peel"].copy()
        umap_pair = ul.merge(up, on=["Cultivar", "Stage"], suffixes=("_leaf", "_peel"), how="inner")
        umap_pair["umap_euclidean_distance"] = np.sqrt(
            (umap_pair["UMAP1_leaf"] - umap_pair["UMAP1_peel"]) ** 2 +
            (umap_pair["UMAP2_leaf"] - umap_pair["UMAP2_peel"]) ** 2
        )

        cmp_df = cent_dist_df.merge(
            umap_pair[["Cultivar", "Stage", "umap_euclidean_distance"]],
            on=["Cultivar", "Stage"],
            how="inner"
        )
        rho_cos, p_cos = spearmanr(
            cmp_df["cosine_distance_original_space"], cmp_df["umap_euclidean_distance"]
        )
        rho_euc, p_euc = spearmanr(
            cmp_df["euclidean_distance_original_space"], cmp_df["umap_euclidean_distance"]
        )
        cmp_df["spearman_rho_cosine_vs_umap"] = rho_cos
        cmp_df["spearman_p_cosine_vs_umap"] = p_cos
        cmp_df["spearman_rho_euclidean_vs_umap"] = rho_euc
        cmp_df["spearman_p_euclidean_vs_umap"] = p_euc
        cmp_path = outdir / "distance_comparison_original_vs_umap.csv"
        cmp_df.to_csv(cmp_path, index=False)

        # scatter
        plt.figure(figsize=(5, 4))
        plt.scatter(
            cmp_df["cosine_distance_original_space"],
            cmp_df["umap_euclidean_distance"],
            alpha=0.8
        )
        plt.xlabel("Cosine distance in original descriptor space")
        plt.ylabel("Euclidean distance in 2D UMAP space")
        plt.title(f"Spearman rho = {rho_cos:.3f}, p = {p_cos:.3g}")
        plt.tight_layout()
        plt.savefig(outdir / "fig_original_vs_umap_distance_scatter.pdf")
        plt.close()

    # 8) figures
    # original-space stage distance (centroid)
    stage_order = sorted(stage_summary_df["Stage"].tolist())
    data_by_stage = [
        cent_dist_df.loc[cent_dist_df["Stage"] == s, "cosine_distance_original_space"].to_numpy()
        for s in stage_order
    ]
    plt.figure(figsize=(5, 4))
    plt.boxplot(data_by_stage, labels=stage_order, showfliers=True)
    plt.ylabel("Cosine distance in original descriptor space")
    plt.xlabel("Stage")
    plt.tight_layout()
    plt.savefig(outdir / "fig_original_space_distance_by_stage.pdf")
    plt.close()

    # retrieval figure
    r = retrieval_df.set_index("task")
    tasks = ["sample_pair", "cultivar_stage_centroid"]
    labels = ["Sample pair", "Cultivar × Stage centroid"]
    metrics = ["top1", "top5", "top10"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    for ax, task, lab in zip(axes, tasks, labels):
        obs_vals = [r.loc[task, m] for m in metrics]
        null_vals = [r.loc[task, f"null_{m}_mean"] for m in metrics]
        ax.bar(x - width/2, obs_vals, width, label="Observed")
        ax.bar(x + width/2, null_vals, width, label="Null mean")
        ax.set_xticks(x)
        ax.set_xticklabels(["Top-1", "Top-5", "Top-10"])
        ax.set_title(lab)
        ax.set_ylabel("Retrieval accuracy")
    axes[0].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outdir / "fig_original_space_retrieval.pdf")
    plt.close()

    # 9) write brief summary text
    summary_txt = outdir / "semantic_original_space_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Original-space descriptor analysis completed.\n")
        f.write(f"Weighting scheme: {args.weighting}\n")
        f.write(f"Matched sample pairs: {pair_df.shape[0]}\n")
        f.write(f"Matched cultivar × stage centroids: {cent_pair.shape[0]}\n\n")
        f.write("Stage summary (centroid cosine distance):\n")
        f.write(stage_summary_df.to_string(index=False))
        f.write("\n\nRetrieval metrics:\n")
        f.write(retrieval_df.to_string(index=False))
        f.write("\n")

    print(f"[OK] wrote: {prof_path}")
    print(f"[OK] wrote: {pair_dist_path}")
    print(f"[OK] wrote: {cent_dist_path}")
    print(f"[OK] wrote: {stage_summary_path}")
    print(f"[OK] wrote: {stage_tests_path}")
    print(f"[OK] wrote: {retrieval_path}")
    print(f"[OK] wrote: {stagewise_path}")
    print(f"[OK] wrote: {summary_txt}")
    print("[DONE] original-space semantic transfer audit completed.")

if __name__ == "__main__":
    main()