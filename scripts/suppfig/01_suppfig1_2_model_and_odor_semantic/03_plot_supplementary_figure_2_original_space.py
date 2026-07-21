#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================
# HARD-LOCKED PALETTE (match Fig6/Fig7 style)
# ============================
C = {
    "g": "#7DC69B",
    "b": "#9BD7F3",
    "g2": "#D5EAD9",
    "b2": "#D8EEFB",
    "p": "#DCD7EB",
    "r": "#F2A1A7",
    "r2": "#FBDDDD",
    "o": "#FCE6CF",
}

STAGE_ORDER = ["S1", "S2", "S3", "S4"]
STAGE_COLOR = {"S1": C["b"], "S2": C["g"], "S3": C["p"], "S4": C["r"]}

def set_pub_style():
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["axes.titleweight"] = "bold"
    mpl.rcParams["axes.labelsize"] = 11
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["legend.fontsize"] = 9
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_both(fig, out_pdf: Path, out_svg: Path):
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")

def fmt_p(p):
    p = float(p)
    if p == 0:
        return "<0.001"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3g}"

# ============================
# Panel a
# Original-space stagewise distance
# ============================
def plot_panel_a(results_dir: Path, outdir: Path):
    df = pd.read_csv(results_dir / "centroid_pair_distances_original_space.csv")
    df["Stage"] = pd.Categorical(df["Stage"], categories=STAGE_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    rng = np.random.default_rng(1)
    data = []
    labels = []

    for st in STAGE_ORDER:
        vals = df.loc[df["Stage"].astype(str).eq(st), "cosine_distance_original_space"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(st)

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True
    )

    for box, st in zip(bp["boxes"], labels):
        box.set_facecolor(STAGE_COLOR[st])
        box.set_alpha(0.25)
        box.set_edgecolor("#333333")
        box.set_linewidth(1.0)

    for median in bp["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.2)

    for whisker in bp["whiskers"]:
        whisker.set_color("#333333")
        whisker.set_linewidth(1.0)

    for cap in bp["caps"]:
        cap.set_color("#333333")
        cap.set_linewidth(1.0)

    for i, (st, vals) in enumerate(zip(labels, data), start=1):
        xj = i + (rng.random(len(vals)) - 0.5) * 0.18
        ax.scatter(
            xj, vals,
            s=11,
            alpha=0.22,
            color=STAGE_COLOR[st],
            edgecolors="none"
        )

    ax.set_xlabel("Stage")
    ax.set_ylabel("Cosine distance in original descriptor space")
    ax.set_title("Supplementary Fig. X a | Stagewise cross-organ distance")

    # optional lightweight stats note
    summ = pd.read_csv(results_dir / "stage_distance_summary.csv")
    txt = []
    for _, r in summ.iterrows():
        txt.append(f"{r['Stage']}: {r['mean_cosine_distance']:.3f}")
    ax.text(
        0.02, 0.98,
        "Mean cosine distance\n" + "\n".join(txt),
        transform=ax.transAxes,
        ha="left", va="top", fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    )

    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()

    save_both(
        fig,
        outdir / "SuppFigX_a_original_space_distance.pdf",
        outdir / "SuppFigX_a_original_space_distance.svg",
    )
    plt.close(fig)

# ============================
# Panel b
# Original-space vs UMAP concordance
# ============================
def plot_panel_b(results_dir: Path, outdir: Path):
    df = pd.read_csv(results_dir / "distance_comparison_original_vs_umap.csv")
    df["Stage"] = pd.Categorical(df["Stage"], categories=STAGE_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    for st in STAGE_ORDER:
        sub = df[df["Stage"].astype(str).eq(st)]
        if sub.empty:
            continue
        ax.scatter(
            sub["cosine_distance_original_space"],
            sub["umap_euclidean_distance"],
            s=36,
            color=STAGE_COLOR[st],
            alpha=0.95,
            edgecolors="#222222",
            linewidths=0.25,
            label=st,
            zorder=3
        )

    # global fit line
    x = df["cosine_distance_original_space"].to_numpy(dtype=float)
    y = df["umap_euclidean_distance"].to_numpy(dtype=float)
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = coef[0] * xs + coef[1]
    ax.plot(xs, ys, color="#444444", lw=1.2, alpha=0.8, zorder=2)

    rho_cos = float(df["spearman_rho_cosine_vs_umap"].iloc[0])
    p_cos = float(df["spearman_p_cosine_vs_umap"].iloc[0])
    rho_euc = float(df["spearman_rho_euclidean_vs_umap"].iloc[0])
    p_euc = float(df["spearman_p_euclidean_vs_umap"].iloc[0])

    stat_txt = (
        f"Cosine vs UMAP: ρ = {rho_cos:.3f}, p = {fmt_p(p_cos)}\n"
        f"Euclidean vs UMAP: ρ = {rho_euc:.3f}, p = {fmt_p(p_euc)}"
    )
    ax.text(
        0.03, 0.97,
        stat_txt,
        transform=ax.transAxes,
        ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    )

    ax.set_xlabel("Cosine distance in original descriptor space")
    ax.set_ylabel("Euclidean distance in 2D UMAP space")
    ax.set_title("Supplementary Fig. X b | Concordance with UMAP distance")
    ax.legend(frameon=False, title="Stage", loc="lower right")
    ax.grid(alpha=0.15)

    fig.tight_layout()
    save_both(
        fig,
        outdir / "SuppFigX_b_original_vs_umap.pdf",
        outdir / "SuppFigX_b_original_vs_umap.svg",
    )
    plt.close(fig)

# ============================
# Panel c
# Original-space retrieval
# ============================
def plot_panel_c(results_dir: Path, outdir: Path):
    df = pd.read_csv(results_dir / "retrieval_metrics_original_space.csv")

    def draw(ax, row, title):
        metrics = ["top1", "top5", "top10"]
        labels = ["Top-1", "Top-5", "Top-10"]
        xs = np.arange(3)

        obs = [row["top1"], row["top5"], row["top10"]]
        null_mu = [row["null_top1_mean"], row["null_top5_mean"], row["null_top10_mean"]]
        null_lo = [row["null_top1_lo"], row["null_top5_lo"], row["null_top10_lo"]]
        null_hi = [row["null_top1_hi"], row["null_top5_hi"], row["null_top10_hi"]]

        ax.bar(xs, obs, width=0.50, color=C["g"], alpha=0.95, edgecolor="#333333", linewidth=0.3)

        for i in range(3):
            x_left, x_right = xs[i] - 0.38, xs[i] + 0.38
            ax.fill_between(
                [x_left, x_right],
                [null_lo[i], null_lo[i]],
                [null_hi[i], null_hi[i]],
                color=C["p"],
                alpha=0.35,
                linewidth=0
            )
            ax.hlines(
                null_mu[i],
                x_left, x_right,
                colors=C["p"],
                linestyles="--",
                linewidth=1.5
            )
            ax.text(xs[i], obs[i], f"{obs[i]:.3f}", ha="center", va="bottom", fontsize=9)

        ptxt = (
            f"p(top1)={fmt_p(row['p_top1'])}\n"
            f"p(top5)={fmt_p(row['p_top5'])}\n"
            f"p(top10)={fmt_p(row['p_top10'])}"
        )
        ax.text(
            0.98, 0.98,
            ptxt,
            transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
        )

        ax.set_xticks(xs, labels)
        ax.set_ylabel("Retrieval accuracy")
        ax.set_title(title)
        ax.set_ylim(0, max(max(obs), max(null_hi)) * 1.30 + 1e-6)

    row_sample = df[df["task"].eq("sample_pair")].iloc[0].to_dict()
    row_cent = df[df["task"].eq("cultivar_stage_centroid")].iloc[0].to_dict()

    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=False)
    draw(axs[0], row_sample, "Sample pair")
    draw(axs[1], row_cent, "Cultivar × Stage centroid")

    obs_proxy = plt.Rectangle((0, 0), 1, 1, fc=C["g"], alpha=0.95, ec="#333333", lw=0.3)
    null_proxy = plt.Line2D([0], [0], color=C["p"], linestyle="--", lw=1.5)
    fig.legend(
        [obs_proxy, null_proxy],
        ["Observed", "Permutation null mean; band = 95% CI"],
        loc="upper center",
        ncol=2,
        frameon=False
    )

    fig.suptitle("Supplementary Fig. X c | Retrieval in original descriptor space", y=0.98, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    save_both(
        fig,
        outdir / "SuppFigX_c_original_space_retrieval.pdf",
        outdir / "SuppFigX_c_original_space_retrieval.svg",
    )
    plt.close(fig)

# ============================
# Optional combined figure
# ============================
def plot_combined(results_dir: Path, outdir: Path):
    stage_df = pd.read_csv(results_dir / "centroid_pair_distances_original_space.csv")
    stage_df["Stage"] = pd.Categorical(stage_df["Stage"], categories=STAGE_ORDER, ordered=True)

    cmp_df = pd.read_csv(results_dir / "distance_comparison_original_vs_umap.csv")
    cmp_df["Stage"] = pd.Categorical(cmp_df["Stage"], categories=STAGE_ORDER, ordered=True)

    ret_df = pd.read_csv(results_dir / "retrieval_metrics_original_space.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    ax1, ax2, ax3 = axes

    # a
    rng = np.random.default_rng(1)
    data, labels = [], []
    for st in STAGE_ORDER:
        vals = stage_df.loc[stage_df["Stage"].astype(str).eq(st), "cosine_distance_original_space"].dropna().to_numpy()
        if len(vals):
            data.append(vals)
            labels.append(st)

    bp = ax1.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for box, st in zip(bp["boxes"], labels):
        box.set_facecolor(STAGE_COLOR[st])
        box.set_alpha(0.25)
        box.set_edgecolor("#333333")
        box.set_linewidth(1.0)
    for median in bp["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.2)
    for i, (st, vals) in enumerate(zip(labels, data), start=1):
        xj = i + (rng.random(len(vals)) - 0.5) * 0.18
        ax1.scatter(xj, vals, s=11, alpha=0.22, color=STAGE_COLOR[st], edgecolors="none")
    ax1.set_xlabel("Stage")
    ax1.set_ylabel("Cosine distance")
    ax1.set_title("a  Original-space stagewise distance")
    ax1.grid(axis="y", alpha=0.15)

    # b
    for st in STAGE_ORDER:
        sub = cmp_df[cmp_df["Stage"].astype(str).eq(st)]
        if sub.empty:
            continue
        ax2.scatter(
            sub["cosine_distance_original_space"],
            sub["umap_euclidean_distance"],
            s=34, color=STAGE_COLOR[st], alpha=0.95,
            edgecolors="#222222", linewidths=0.25, label=st
        )
    x = cmp_df["cosine_distance_original_space"].to_numpy(dtype=float)
    y = cmp_df["umap_euclidean_distance"].to_numpy(dtype=float)
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax2.plot(xs, coef[0] * xs + coef[1], color="#444444", lw=1.2, alpha=0.8)
    rho = float(cmp_df["spearman_rho_cosine_vs_umap"].iloc[0])
    p = float(cmp_df["spearman_p_cosine_vs_umap"].iloc[0])
    ax2.text(
        0.03, 0.97,
        f"ρ = {rho:.3f}\np = {fmt_p(p)}",
        transform=ax2.transAxes,
        ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    )
    ax2.set_xlabel("Original-space cosine distance")
    ax2.set_ylabel("2D UMAP distance")
    ax2.set_title("b  Concordance with UMAP")
    ax2.legend(frameon=False, title="Stage", loc="lower right")
    ax2.grid(alpha=0.15)

    # c
    row_cent = ret_df[ret_df["task"].eq("cultivar_stage_centroid")].iloc[0].to_dict()
    xs = np.arange(3)
    obs = [row_cent["top1"], row_cent["top5"], row_cent["top10"]]
    null_mu = [row_cent["null_top1_mean"], row_cent["null_top5_mean"], row_cent["null_top10_mean"]]
    null_lo = [row_cent["null_top1_lo"], row_cent["null_top5_lo"], row_cent["null_top10_lo"]]
    null_hi = [row_cent["null_top1_hi"], row_cent["null_top5_hi"], row_cent["null_top10_hi"]]

    ax3.bar(xs, obs, width=0.50, color=C["g"], alpha=0.95, edgecolor="#333333", linewidth=0.3)
    for i in range(3):
        x_left, x_right = xs[i] - 0.38, xs[i] + 0.38
        ax3.fill_between([x_left, x_right], [null_lo[i], null_lo[i]], [null_hi[i], null_hi[i]], color=C["p"], alpha=0.35, linewidth=0)
        ax3.hlines(null_mu[i], x_left, x_right, colors=C["p"], linestyles="--", linewidth=1.5)
    ax3.set_xticks(xs, ["Top-1", "Top-5", "Top-10"])
    ax3.set_ylabel("Retrieval accuracy")
    ax3.set_title("c  Original-space centroid retrieval")
    ax3.text(
        0.98, 0.98,
        f"p(top1)={fmt_p(row_cent['p_top1'])}\n"
        f"p(top5)={fmt_p(row_cent['p_top5'])}\n"
        f"p(top10)={fmt_p(row_cent['p_top10'])}",
        transform=ax3.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    )

    fig.suptitle("Supplementary Fig. X | Original-space validation of cross-organ odor-semantic transfer", y=0.99, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_both(
        fig,
        outdir / "SuppFigX_combined.pdf",
        outdir / "SuppFigX_combined.svg",
    )
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        type=str,
        default="results/supplementary_figure_2_original_space/original_space_transfer"
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="results/supplementary_figure_2_original_space/figure"
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    outdir = Path(args.outdir).resolve()
    ensure_dir(outdir)
    set_pub_style()

    plot_panel_a(results_dir, outdir)
    plot_panel_b(results_dir, outdir)
    plot_panel_c(results_dir, outdir)
    plot_combined(results_dir, outdir)

    print("[OK] Saved Supplementary Fig. X panels to:", outdir)

if __name__ == "__main__":
    main()