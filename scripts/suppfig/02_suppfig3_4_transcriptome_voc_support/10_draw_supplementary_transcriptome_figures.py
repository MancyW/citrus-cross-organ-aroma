#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw Supplementary Fig. 3 and Supplementary Fig. 4 for the
leaf-peel transcriptome/VOC support analysis.

Usage:
    python draw_supplementary_transcriptome_figures.py \
        --base supplementary_transcriptome_support \
        --out figures/supplementary_transcriptome

The script expects the directory structure created by
supplementary_transcriptome_support.tar.gz.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# Global style
# ============================================================

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.linewidth"] = 0.75
mpl.rcParams["xtick.major.width"] = 0.75
mpl.rcParams["ytick.major.width"] = 0.75
mpl.rcParams["xtick.major.size"] = 3
mpl.rcParams["ytick.major.size"] = 3
mpl.rcParams["axes.titlesize"] = 8.5
mpl.rcParams["axes.labelsize"] = 8
mpl.rcParams["xtick.labelsize"] = 7
mpl.rcParams["ytick.labelsize"] = 7
mpl.rcParams["legend.fontsize"] = 7

PALETTE = ["#7DC69B", "#9BD7F3", "#D5EAD9", "#D8EEFB", "#DCD7EB", "#F2A1A7", "#FBDDDD", "#FCE6CF"]

# Figure canvas sizes. A4 width = 210 mm = 8.2677 inches.
# Use fixed canvas sizes and avoid bbox_inches="tight" so that PDF/SVG/PNG
# retain an A4-width artboard for downstream editing in Adobe Illustrator.
A4_WIDTH_IN = 210 / 25.4
FIG3_HEIGHT_IN = 7.15
FIG4_HEIGHT_IN = 8.10
ORGAN_COLORS = {"Leaf": PALETTE[0], "Peel": PALETTE[5]}
STAGE_MARKERS = {"S1": "o", "S2": "^", "S3": "s", "S4": "D"}
STAGE_ORDER = ["S1", "S2", "S3", "S4"]
STAGE_COLORS = {"S1": PALETTE[0], "S2": PALETTE[1], "S3": PALETTE[4], "S4": PALETTE[5]}
ORGAN_MARKERS = {"Leaf": "o", "Peel": "^"}

# A restrained diverging map for correlation heatmaps.
CORR_CMAP = LinearSegmentedColormap.from_list(
    "main_palette_div", [PALETTE[1], "#FFFFFF", PALETTE[5]], N=256
)
PATHWAY_CMAP = LinearSegmentedColormap.from_list(
    "main_palette_seq", ["#FFFFFF", PALETTE[2], PALETTE[0]], N=256
)


# ============================================================
# Helper functions
# ============================================================

def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t")


def clean_module_name(x: str) -> str:
    x = str(x)
    x = x.replace("LeafME__", "L_").replace("PeelME__", "P_")
    return x


def clean_voc_name(x: str) -> str:
    x = str(x)
    x = x.replace("PeelVOC__", "")
    x = x.replace("E_beta_Ocimene", "(E)-β-Ocimene")
    x = x.replace("beta_Myrcene", "β-Myrcene")
    x = x.replace("D_Limonene", "D-Limonene")
    x = x.replace("_", " ")
    return x


def pretty_gene_set_name(x: str) -> str:
    mapping = {
        "broad_voc_metabolism_plus_tf": "VOC metabolism + TF",
        "broad_voc_metabolism_related": "VOC metabolism",
        "broad_terpenoid_related": "Terpenoid-related",
        "go_transcription_factor": "Transcription factor",
        "go_oxidoreductase": "Oxidoreductase",
        "go_glycosyltransferase": "Glycosyltransferase",
        "go_alcohol_aldehyde_lipoxygenase_related": "Aldehyde/alcohol/LOX",
        "kegg_fatty_acid_related": "Fatty acid-related",
        "kegg_terpenoid_backbone": "Terpenoid backbone",
        "kegg_carotenoid": "Carotenoid",
        "kegg_diterpenoid": "Diterpenoid",
        "kegg_sesquiterpenoid_triterpenoid": "Sesqui-/triterpenoid",
        "kegg_phenylpropanoid": "Phenylpropanoid",
    }
    return mapping.get(str(x), str(x))


def significance_star(q: float) -> str:
    if pd.isna(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def add_panel_label(ax, label: str, x: float = -0.15, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
    )


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, outbase: Path) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        dpi = 600 if ext == "png" else None
        fig.savefig(outbase.with_suffix(f".{ext}"), bbox_inches=None, pad_inches=0, dpi=dpi)
    plt.close(fig)


def parse_permanova_terms(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Permutation") or line.startswith("Terms") or line.startswith("adonis"):
                continue
            if line.startswith("---") or line.startswith("Signif"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) >= 6 and parts[0] not in {"Df", "Residual", "Total"}:
                # term Df SumOfSqs R2 F p ...
                try:
                    term = parts[0]
                    r2 = float(parts[3])
                    pval = float(parts[5])
                    rows.append({"term": term, "R2": r2, "p_value": pval})
                except Exception:
                    pass
    df = pd.DataFrame(rows)
    label_map = {
        "organ": "Organ",
        "cultivar": "Cultivar",
        "stage": "Stage",
        "organ:stage": "Organ × stage",
    }
    df["label"] = df["term"].map(label_map).fillna(df["term"])
    return df


def draw_heatmap(
    ax,
    matrix: pd.DataFrame,
    annot: pd.DataFrame | None = None,
    cmap=CORR_CMAP,
    vmin=-1,
    vmax=1,
    cbar_label="Spearman ρ",
    cbar=True,
    fontsize=6,
    rotate_x=45,
) -> None:
    im = ax.imshow(matrix.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=rotate_x, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annot is not None:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.iat[i, j]
                if pd.notna(val):
                    text = annot.iat[i, j]
                    txt_color = "white" if abs(val) > (0.65 if vmax == 1 else 0.55 * vmax) else "#333333"
                    ax.text(j, i, text, ha="center", va="center", fontsize=fontsize, color=txt_color)

    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(cbar_label, fontsize=7)
        cb.ax.tick_params(labelsize=6)


# ============================================================
# Supplementary Fig. 3
# ============================================================

def draw_supplementary_fig3(base: Path, outdir: Path) -> None:
    """Draw Supplementary Fig. 3 using the split-PCA design.

    Design logic:
    a, PCA colored by organ confirms organ-distinct transcriptomes.
    b, The same PCA colored by stage makes developmental structure visible.
    c, PERMANOVA quantifies organ, cultivar, stage and organ-by-stage effects.
    d, Stage-specific gene-set Mantel heatmap summarizes pathway-level
       cross-organ coordination.
    """
    pca = read_tsv(base / "analysis/rnaseq_qc/PCA_scores.tsv")
    pcvar = read_tsv(base / "analysis/rnaseq_qc/PCA_percent_variance.tsv")
    permanova = parse_permanova_terms(base / "analysis/rnaseq_qc/PERMANOVA_by_terms_all_samples.txt")
    stage_mantel = read_tsv(base / "analysis/pathway_coordination/pathway_gene_set_mantel_by_stage.tsv")

    pc1 = float(pcvar.loc[pcvar["PC"] == "PC1", "percent_variance"].iloc[0])
    pc2 = float(pcvar.loc[pcvar["PC"] == "PC2", "percent_variance"].iloc[0])

    fig = plt.figure(figsize=(A4_WIDTH_IN, FIG3_HEIGHT_IN))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.25],
        wspace=0.42,
        hspace=0.58,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # A. PCA colored by organ
    for organ in ["Leaf", "Peel"]:
        sub = pca[pca["organ"] == organ]
        ax_a.scatter(
            sub["PC1"], sub["PC2"],
            s=20,
            c=ORGAN_COLORS[organ],
            marker="o",
            alpha=0.72,
            linewidths=0.35,
            edgecolors="white",
            label=organ,
        )
    ax_a.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax_a.set_ylabel(f"PC2 ({pc2:.1f}%)")
    ax_a.axhline(0, color="#DCD7EB", lw=0.7, zorder=0)
    ax_a.axvline(0, color="#DCD7EB", lw=0.7, zorder=0)
    ax_a.legend(title="Organ", frameon=False, loc="upper left", handletextpad=0.4, borderaxespad=0.2)
    despine(ax_a)
    add_panel_label(ax_a, "a")
    ax_a.set_title("PCA colored by organ", loc="left", fontsize=8.5, pad=6)

    # B. PCA colored by developmental stage. Organ is retained as marker shape.
    for stage in STAGE_ORDER:
        for organ in ["Leaf", "Peel"]:
            sub = pca[(pca["stage"] == stage) & (pca["organ"] == organ)]
            ax_b.scatter(
                sub["PC1"], sub["PC2"],
                s=22,
                c=STAGE_COLORS[stage],
                marker=ORGAN_MARKERS[organ],
                alpha=0.78,
                linewidths=0.4,
                edgecolors="#333333",
            )
    ax_b.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax_b.set_ylabel(f"PC2 ({pc2:.1f}%)")
    ax_b.axhline(0, color="#DCD7EB", lw=0.7, zorder=0)
    ax_b.axvline(0, color="#DCD7EB", lw=0.7, zorder=0)
    stage_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=STAGE_COLORS[s],
               markeredgecolor="#333333", label=s, markersize=6, linestyle="none")
        for s in STAGE_ORDER
    ]
    # Stage legend only. Organ is encoded by point shape (circle = leaf, triangle = peel)
    # and described in the figure legend to keep this panel uncluttered.
    ax_b.legend(handles=stage_handles, title="Stage", loc="upper left", frameon=False,
                handletextpad=0.4, borderaxespad=0.2)
    despine(ax_b)
    add_panel_label(ax_b, "b")
    ax_b.set_title("PCA colored by developmental stage", loc="left", fontsize=8.5, pad=6)

    # C. PERMANOVA bars
    order = ["Organ", "Cultivar", "Stage", "Organ × stage"]
    dfb = permanova.set_index("label").reindex(order).reset_index()
    bar_colors = [PALETTE[0], PALETTE[1], PALETTE[4], PALETTE[5]]
    bars = ax_c.bar(np.arange(len(dfb)), dfb["R2"], color=bar_colors, edgecolor="#666666", linewidth=0.6)
    ax_c.set_xticks(np.arange(len(dfb)))
    ax_c.set_xticklabels(dfb["label"], rotation=35, ha="right")
    ax_c.set_ylabel("Variance explained (R²)")
    ax_c.set_ylim(0, max(0.30, dfb["R2"].max() * 1.25))
    for bar, p in zip(bars, dfb["p_value"]):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            "p=0.001",
            ha="center",
            va="bottom",
            fontsize=6.2,
            rotation=90,
        )
    despine(ax_c)
    add_panel_label(ax_c, "c")
    ax_c.set_title("PERMANOVA of transcriptomes", loc="left", fontsize=8.5, pad=6)

    # D. Stage-specific pathway Mantel heatmap
    preferred_sets = [
        "broad_voc_metabolism_plus_tf",
        "broad_voc_metabolism_related",
        "broad_terpenoid_related",
        "kegg_terpenoid_backbone",
        "kegg_sesquiterpenoid_triterpenoid",
        "kegg_fatty_acid_related",
        "go_alcohol_aldehyde_lipoxygenase_related",
        "go_oxidoreductase",
        "go_glycosyltransferase",
        "go_transcription_factor",
        "kegg_carotenoid",
        "kegg_phenylpropanoid",
    ]
    st = stage_mantel[stage_mantel["gene_set"].isin(preferred_sets)].copy()
    st["gene_set_label"] = st["gene_set"].map(pretty_gene_set_name)
    st["stage"] = pd.Categorical(st["stage"], STAGE_ORDER, ordered=True)
    st["gene_set_label"] = pd.Categorical(
        st["gene_set_label"],
        [pretty_gene_set_name(x) for x in preferred_sets],
        ordered=True,
    )
    mat = st.pivot(index="gene_set_label", columns="stage", values="mantel_r")
    qmat = st.pivot(index="gene_set_label", columns="stage", values="padj_BH")
    ann = mat.copy().astype(object)
    for i in mat.index:
        for j in mat.columns:
            q = qmat.loc[i, j]
            ann.loc[i, j] = f"{mat.loc[i, j]:.2f}{significance_star(q)}"
    draw_heatmap(
        ax_d,
        mat,
        ann,
        cmap=PATHWAY_CMAP,
        vmin=0,
        vmax=0.85,
        cbar_label="Mantel r",
        fontsize=5.4,
        rotate_x=0,
    )
    add_panel_label(ax_d, "d")
    ax_d.set_title("Pathway-level cross-organ coordination", loc="left", fontsize=8.5, pad=6)

    fig.text(
        0.01,
        0.005,
        "Stage-complete transcriptomic subset: 13 cultivars; JG excluded because S1 RNA-seq samples were unavailable.",
        fontsize=6.2,
        ha="left",
        va="bottom",
        color="#666666",
    )
    save_figure(fig, outdir / "Supplementary_Fig3_cross_organ_transcriptomic_coordination")


# ============================================================
# Supplementary Fig. 4
# ============================================================

def choose_top_pairs_for_matrix(pairs: pd.DataFrame, n_pairs: int = 18) -> Tuple[List[str], List[str]]:
    top = pairs.sort_values("abs_cor", ascending=False).head(n_pairs)
    leaf_modules = list(dict.fromkeys(top["left"].tolist()))
    peel_modules = list(dict.fromkeys(top["right"].tolist()))
    return leaf_modules, peel_modules


def draw_chain_panel(ax, chains: pd.DataFrame, max_chains: int = 8) -> None:
    # Select one high-scoring chain per VOC, then rank by evidence.
    selected = (
        chains.sort_values(["voc_priority", "evidence_score"], ascending=[True, False])
        .groupby("peel_voc_original", as_index=False)
        .head(1)
        .sort_values("evidence_score", ascending=False)
        .head(max_chains)
        .reset_index(drop=True)
    )
    # Reorder by VOC priority to make the panel easier to read.
    selected = selected.sort_values("voc_priority").reset_index(drop=True)

    ax.set_axis_off()
    x_leaf, x_peel, x_voc = 0.12, 0.50, 0.88
    n = len(selected)
    y_positions = np.linspace(0.90, 0.12, n)

    ax.text(x_leaf, 0.99, "Leaf module", ha="center", va="top", fontsize=7.5, fontweight="bold", transform=ax.transAxes)
    ax.text(x_peel, 0.99, "Peel module", ha="center", va="top", fontsize=7.5, fontweight="bold", transform=ax.transAxes)
    ax.text(x_voc, 0.99, "Peel VOC", ha="center", va="top", fontsize=7.5, fontweight="bold", transform=ax.transAxes)

    for y, row in zip(y_positions, selected.itertuples(index=False)):
        leaf = clean_module_name(row.leaf_module)
        peel = clean_module_name(row.peel_module)
        voc = row.peel_voc_original
        ev = float(row.evidence_score)
        lp_cor = float(row.leaf_peel_module_cor)
        pv_cor = float(row.peel_module_peel_voc_cor)
        lw = 0.8 + 4.0 * (ev / selected["evidence_score"].max())
        color1 = "#B65A5A" if lp_cor > 0 else "#3B6EA8"
        color2 = "#B65A5A" if pv_cor > 0 else "#3B6EA8"

        for x, label in [(x_leaf, leaf), (x_peel, peel), (x_voc, voc)]:
            ax.add_patch(Circle((x, y), 0.032, transform=ax.transAxes, facecolor="#FFFFFF", edgecolor="#666666", lw=0.7, zorder=3))
            ax.text(x, y, label, ha="center", va="center", fontsize=5.6, transform=ax.transAxes, zorder=4)

        ax.add_patch(FancyArrowPatch((x_leaf + 0.04, y), (x_peel - 0.04, y), transform=ax.transAxes,
                                     arrowstyle="-", lw=lw, color=color1, alpha=0.85, zorder=1))
        ax.add_patch(FancyArrowPatch((x_peel + 0.04, y), (x_voc - 0.04, y), transform=ax.transAxes,
                                     arrowstyle="-", lw=lw, color=color2, alpha=0.85, zorder=1))
        ax.text(0.50, y - 0.045, f"score={ev:.2f}", ha="center", va="top", fontsize=5.2, color="#666666", transform=ax.transAxes)

    # Sign legend
    ax.plot([0.10, 0.18], [0.035, 0.035], transform=ax.transAxes, color=PALETTE[5], lw=2)
    ax.text(0.20, 0.035, "positive", transform=ax.transAxes, va="center", fontsize=6)
    ax.plot([0.38, 0.46], [0.035, 0.035], transform=ax.transAxes, color=PALETTE[1], lw=2)
    ax.text(0.48, 0.035, "negative", transform=ax.transAxes, va="center", fontsize=6)


def draw_triplet_panel(ax, triplets: pd.DataFrame, max_rows: int = 9) -> None:
    # One representative triplet per VOC, prioritized by evidence score.
    df = (
        triplets.sort_values(["voc_priority", "evidence_score"], ascending=[True, False])
        .groupby("peel_voc_original", as_index=False)
        .head(1)
        .sort_values("evidence_score", ascending=True)
        .tail(max_rows)
        .reset_index(drop=True)
    )
    y = np.arange(len(df))
    colors = ["#B65A5A" if "Tier 1" in str(t) else "#8A8A8A" for t in df["evidence_tier"]]
    sizes = 70 + 260 * (df["evidence_score"] / df["evidence_score"].max())
    ax.scatter(df["evidence_score"], y, s=sizes, c=colors, edgecolors="white", linewidths=0.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["peel_voc_original"])
    ax.set_xlabel("Triplet evidence score")
    ax.grid(axis="x", color="#DCD7EB", lw=0.7)
    despine(ax)

    xmax = max(0.75, df["evidence_score"].max() * 1.95)
    ax.set_xlim(0, xmax)
    for yi, row in enumerate(df.itertuples(index=False)):
        label = f"{row.leaf_gene} → {row.peel_gene}"
        ax.text(row.evidence_score + xmax * 0.025, yi, label, va="center", ha="left", fontsize=5.2)

    # Compact legend
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE[5], markeredgecolor="white", label="Tier 1", markersize=6),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE[4], markeredgecolor="white", label="Tier 2/other", markersize=6),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=6.5)


def draw_supplementary_fig4(base: Path, outdir: Path) -> None:
    module_pairs = read_tsv(base / "analysis/final_evidence_package/top_leaf_peel_module_pairs.tsv")
    module_voc = read_tsv(base / "analysis/module_trait_association/module_vs_peel_key_voc_spearman_overall.tsv")
    chains = read_tsv(base / "analysis/final_figure_package/selected_candidate_module_voc_chains_for_figure.tsv")
    triplets = read_tsv(base / "analysis/final_figure_package/selected_candidate_gene_triplets_for_figure.tsv")

    fig = plt.figure(figsize=(A4_WIDTH_IN, FIG4_HEIGHT_IN))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.10], height_ratios=[1.0, 1.0], wspace=0.42, hspace=0.48)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # A. Selected leaf-peel module correlation matrix
    leaf_mods, peel_mods = choose_top_pairs_for_matrix(module_pairs, n_pairs=18)
    pair_mat = pd.DataFrame(np.nan, index=[clean_module_name(x) for x in leaf_mods], columns=[clean_module_name(x) for x in peel_mods])
    q_mat = pair_mat.copy()
    mp_lookup = module_pairs.set_index(["left", "right"])
    for l in leaf_mods:
        for p in peel_mods:
            if (l, p) in mp_lookup.index:
                pair_mat.loc[clean_module_name(l), clean_module_name(p)] = mp_lookup.loc[(l, p), "cor"]
                q_mat.loc[clean_module_name(l), clean_module_name(p)] = mp_lookup.loc[(l, p), "padj_BH"]
    ann = pair_mat.copy().astype(object)
    for i in pair_mat.index:
        for j in pair_mat.columns:
            val = pair_mat.loc[i, j]
            ann.loc[i, j] = "" if pd.isna(val) else f"{val:.2f}\n{significance_star(q_mat.loc[i, j])}"
    draw_heatmap(ax_a, pair_mat, ann, cmap=CORR_CMAP, vmin=-0.9, vmax=0.9, cbar_label="Spearman ρ", fontsize=5.2, rotate_x=45)
    ax_a.set_title("Selected leaf–peel module correlations", loc="left", fontsize=8.5, pad=6)
    add_panel_label(ax_a, "a")

    # B. Selected modules vs key peel VOCs
    selected_modules = list(dict.fromkeys(chains["leaf_module"].tolist() + chains["peel_module"].tolist()))
    # Keep modules appearing in strong selected chains, but limit display.
    if len(selected_modules) > 12:
        selected_modules = selected_modules[:12]
    selected_vocs = list(dict.fromkeys(chains.sort_values("voc_priority")["peel_voc_original"].tolist()))
    if len(selected_vocs) > 9:
        selected_vocs = selected_vocs[:9]
    # The comprehensive table may not contain a trait_original column.
    # Create a display name from the trait identifier when needed.
    module_voc = module_voc.copy()
    if "trait_original" not in module_voc.columns:
        module_voc["trait_original"] = module_voc["right"].map(clean_voc_name)
    mv = module_voc[module_voc["left"].isin(selected_modules) & module_voc["trait_original"].isin(selected_vocs)].copy()
    mv_mat = pd.DataFrame(np.nan, index=[clean_module_name(x) for x in selected_modules], columns=selected_vocs)
    mv_q = mv_mat.copy()
    lookup = mv.set_index(["left", "trait_original"])
    for m in selected_modules:
        for voc in selected_vocs:
            if (m, voc) in lookup.index:
                mv_mat.loc[clean_module_name(m), voc] = lookup.loc[(m, voc), "cor"]
                mv_q.loc[clean_module_name(m), voc] = lookup.loc[(m, voc), "padj_BH"]
    mv_ann = mv_mat.copy().astype(object)
    for i in mv_mat.index:
        for j in mv_mat.columns:
            val = mv_mat.loc[i, j]
            mv_ann.loc[i, j] = "" if pd.isna(val) else f"{val:.2f}\n{significance_star(mv_q.loc[i, j])}"
    draw_heatmap(ax_b, mv_mat, mv_ann, cmap=CORR_CMAP, vmin=-0.9, vmax=0.9, cbar_label="Spearman ρ", fontsize=4.8, rotate_x=45)
    ax_b.set_title("Module associations with peel VOC traits", loc="left", fontsize=8.5, pad=6)
    add_panel_label(ax_b, "b")

    # C. Candidate module-VOC chains
    draw_chain_panel(ax_c, chains, max_chains=8)
    ax_c.set_title("Representative module–VOC chains", loc="left", fontsize=8.5, pad=6)
    add_panel_label(ax_c, "c", x=-0.04, y=1.04)

    # D. Candidate gene triplets
    draw_triplet_panel(ax_d, triplets, max_rows=9)
    ax_d.set_title("Representative hub gene triplets", loc="left", fontsize=8.5, pad=6)
    add_panel_label(ax_d, "d")

    fig.text(
        0.01,
        0.005,
        "Candidate module–VOC chains and hub gene triplets are association-level evidence and should not be interpreted as validated regulatory links.",
        fontsize=6.2,
        ha="left",
        va="bottom",
        color="#666666",
    )
    save_figure(fig, outdir / "Supplementary_Fig4_module_voc_candidate_support")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Draw transcriptome/VOC support supplementary figures.")
    parser.add_argument("--base", type=str, default=".", help="Base directory of the reviewer1 transcriptome package.")
    parser.add_argument("--out", type=str, default="figures/supplementary_transcriptome", help="Output directory.")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    outdir = Path(args.out)
    if not outdir.is_absolute():
        outdir = base / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    draw_supplementary_fig3(base, outdir)
    draw_supplementary_fig4(base, outdir)

    print("Done. Figures written to:")
    print(f"  {outdir}")
    for name in [
        "Supplementary_Fig3_cross_organ_transcriptomic_coordination.pdf",
        "Supplementary_Fig4_module_voc_candidate_support.pdf",
    ]:
        print(f"  - {outdir / name}")


if __name__ == "__main__":
    main()
