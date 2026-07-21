from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

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

def assign_region_by_quadrant(x: np.ndarray, y: np.ndarray):

    x0 = np.median(x)

    y0 = np.median(y)

    region = np.empty(len(x), dtype=object)

    region[(x >= x0) & (y >= y0)] = "fruity"

    region[(x <  x0) & (y >= y0)] = "woody"

    region[(x <  x0) & (y <  y0)] = "herbal"

    region[(x >= x0) & (y <  y0)] = "floral"

    return region, x0, y0

def star_from_q(q: float) -> str:

    if q < 1e-4: return "****"

    if q < 1e-3: return "***"

    if q < 1e-2: return "**"

    if q < 5e-2: return "*"

    return ""

def plot_fig7a(root: Path, outdir: Path):

    mid = root / "intermediate"

    voc = pd.read_csv(mid / "voc_pom_2d.csv")

    assert all(k in voc.columns for k in ["VOC","x","y"]), "voc_pom_2d.csv missing VOC/x/y"

    fam_col = "Family" if "Family" in voc.columns else None

    if fam_col:

        voc[fam_col] = voc[fam_col].fillna("Unknown")

        top_fams = voc[fam_col].value_counts().head(6).index.tolist()

        voc["FamilyCollapsed"] = np.where(voc[fam_col].isin(top_fams), voc[fam_col], "Other")

    else:

        voc["FamilyCollapsed"] = "VOC"

    cats = sorted(voc["FamilyCollapsed"].unique().tolist())

    palette_cycle = [C["g"], C["b"], C["p"], C["r"], C["o"], C["g2"], C["b2"], C["r2"]]

    cat_color = {cat: palette_cycle[i % len(palette_cycle)] for i, cat in enumerate(cats)}

    x = voc["x"].to_numpy()

    y = voc["y"].to_numpy()

    region, x0, y0 = assign_region_by_quadrant(x, y)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    for cat in cats:

        sub = voc[voc["FamilyCollapsed"].eq(cat)]

        ax.scatter(

            sub["x"], sub["y"],

            s=26,

            color=cat_color[cat],

            alpha=0.95,

            edgecolors="#222222",

            linewidths=0.20,

            label=cat

        )

    ax.axvline(x0, lw=0.9, alpha=0.15, color="#333333")

    ax.axhline(y0, lw=0.9, alpha=0.15, color="#333333")

    for rn in ["fruity","woody","herbal","floral"]:

        m = (region == rn)

        if m.sum() < 5:

            continue

        cx, cy = float(np.mean(x[m])), float(np.mean(y[m]))

        ax.text(

            cx, cy, rn,

            fontsize=11, ha="center", va="center",

            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75)

        )

    ax.set_xlabel("UMAP-1 (odor-descriptor space)")

    ax.set_ylabel("UMAP-2 (odor-descriptor space)")

    ax.set_title("Fig6A | VOC atlas in perceptual space (family-coded)")

    ax.legend(title="VOC family (collapsed)", frameon=False, loc="best")

    fig.tight_layout()

    save_both(fig, outdir / "Fig6A_final.pdf", outdir / "Fig6A_final.svg")

    plt.close(fig)

def plot_fig7b(root: Path, outdir: Path):

    mid = root / "intermediate"

    df = pd.read_csv(mid / "sample_desc_2d_relative_v4.csv")

    need = ["PairID","Cultivar","Stage","Rep","Organ","x","y"]

    for c in need:

        assert c in df.columns, f"Missing {c} in sample_desc_2d_relative_v4.csv"

    df["Stage"] = pd.Categorical(df["Stage"], categories=STAGE_ORDER, ordered=True)

    leaf = df[df["Organ"].eq("Leaf")].copy()

    peel = df[df["Organ"].eq("Peel")].copy()

    pair = leaf.merge(peel[["PairID","x","y"]], on="PairID", how="inner", suffixes=("_leaf","_peel"))

    pair["dx"] = pair["x_peel"] - pair["x_leaf"]

    pair["dy"] = pair["y_peel"] - pair["y_leaf"]

    pair["dist"] = np.sqrt(pair["dx"]**2 + pair["dy"]**2)

    cent = df.groupby(["Organ","Cultivar","Stage"], as_index=False)[["x","y"]].mean()

    cent_leaf = cent[cent["Organ"].eq("Leaf")].copy()

    cent_peel = cent[cent["Organ"].eq("Peel")].copy()

    cent_pair = cent_leaf.merge(

        cent_peel[["Cultivar","Stage","x","y"]],

        on=["Cultivar","Stage"],

        how="inner",

        suffixes=("_leaf","_peel")

    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8), gridspec_kw={"width_ratios":[1.25,1.0]})

    ax, ax2 = axes

    ax.scatter(df["x"], df["y"], s=10, color="#777777", alpha=0.18, edgecolors="none", zorder=1)

    for organ, marker in [("Leaf","o"),("Peel","^")]:

        sub = cent[cent["Organ"].eq(organ)]

        colors = sub["Stage"].astype(str).map(STAGE_COLOR)

        ax.scatter(

            sub["x"], sub["y"],

            s=40,

            marker=marker,

            c=list(colors),

            alpha=0.95,

            edgecolors="#222222",

            linewidths=0.35,

            zorder=3

        )

    for _, r in cent_pair.iterrows():

        col = STAGE_COLOR.get(str(r["Stage"]), C["p"])

        ax.annotate(

            "", xy=(r["x_peel"], r["y_peel"]), xytext=(r["x_leaf"], r["y_leaf"]),

            arrowprops=dict(arrowstyle='-|>', lw=1.1, color=col, alpha=0.90, mutation_scale=10),

            zorder=4

        )

    ax.set_xlabel("UMAP-1 (relative-weighted odor-descriptor space)")

    ax.set_ylabel("UMAP-2 (relative-weighted odor-descriptor space)")

    ax.set_title("Fig6B | Cross-organ perceptual transfer (centroid arrows)")

    organ_handles = [

        plt.Line2D([0],[0], marker='o', linestyle='', label='Leaf centroid', markersize=7, color="#222222"),

        plt.Line2D([0],[0], marker='^', linestyle='', label='Peel centroid', markersize=7, color="#222222"),

        plt.Line2D([0],[0], marker='o', linestyle='', label='All samples (background)', markersize=6, color="#777777", alpha=0.7),

    ]

    stage_handles = [plt.Line2D([0],[0], marker='s', linestyle='', label=s, markersize=7, color=STAGE_COLOR[s]) for s in STAGE_ORDER]

    leg1 = ax.legend(handles=organ_handles, frameon=False, loc="upper right", title="Layers")

    ax.add_artist(leg1)

    ax.legend(handles=stage_handles, frameon=False, loc="lower right", title="Stage")

    rng = np.random.default_rng(1)

    data, labels = [], []

    for st in STAGE_ORDER:

        vals = pair.loc[pair["Stage"].astype(str).eq(st), "dist"].dropna().to_numpy()

        if len(vals):

            data.append(vals); labels.append(st)

    bp = ax2.boxplot(data, labels=labels, showfliers=False, patch_artist=True)

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

        ax2.scatter(xj, vals, s=9, alpha=0.22, color=STAGE_COLOR[st])

    ax2.set_xlabel("Stage")

    ax2.set_ylabel("Euclidean distance in 2D UMAP space (a.u.)")

    ax2.set_title("Displacement magnitude by stage")

    fig.tight_layout()

    save_both(fig, outdir / "Fig6B_final.pdf", outdir / "Fig6B_final.svg")

    plt.close(fig)

def plot_fig7c(root: Path, outdir: Path):

    res = root / "results"

    stat = pd.read_csv(res / "Fig6C_descriptor_shift_stats_relative_v4.csv")

    topk = 15

    sel = stat.reindex(stat["mean_delta"].abs().sort_values(ascending=False).index).head(topk).copy()

    sel = sel.sort_values("mean_delta")

    y = np.arange(len(sel))

    colors = [C["r"] if v < 0 else C["g"] for v in sel["mean_delta"].to_numpy()]

    fig, ax = plt.subplots(figsize=(7.6, 6.1))

    ax.barh(y, sel["mean_delta"].to_numpy(), color=colors, alpha=0.95)

    xerr_left = sel["mean_delta"].to_numpy() - sel["ci_lo"].to_numpy()

    xerr_right = sel["ci_hi"].to_numpy() - sel["mean_delta"].to_numpy()

    ax.errorbar(sel["mean_delta"].to_numpy(), y, xerr=[xerr_left, xerr_right],

                fmt="none", capsize=3, lw=1, color="#222222")

    ax.axvline(0, lw=1, color="#333333")

    ax.set_yticks(y, sel["descriptor"].tolist())

    ax.set_xlabel("Mean(Peel − Leaf) in odor-descriptor probability (paired by tree; relative-weighted)")

    ax.set_title(f"Fig6C | Descriptor shift with 95% bootstrap CI (n={int(sel['n_pair'].iloc[0])})")

    for i, (_, r) in enumerate(sel.iterrows()):

        s = star_from_q(float(r["q_value"]))

        if s:

            xpos = float(r["ci_hi"]) if float(r["mean_delta"]) >= 0 else float(r["ci_lo"])

            ax.text(xpos, i, f"  {s}", va="center", fontsize=10)

    ax.text(

        0.01, 0.02,

        "Significance (BH-FDR q):  *<0.05,  **<0.01,  ***<0.001,  ****<1e-4",

        transform=ax.transAxes, fontsize=9, va="bottom", ha="left",

        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.80)

    )

    fig.tight_layout()

    save_both(fig, outdir / "Fig6C_final.pdf", outdir / "Fig6C_final.svg")

    plt.close(fig)

def plot_fig7d(root: Path, outdir: Path):

    res = root / "results"

    met = pd.read_csv(res / "Fig6D_metrics_relative_main_v4.csv")

    r1 = pd.read_csv(res / "Fig6D_ranks_sample_pair_relative_v4.csv")["rank"].to_numpy()

    r2 = pd.read_csv(res / "Fig6D_ranks_centroid_relative_v4.csv")["rank"].to_numpy()

    def fmt_p(p):

        p = float(p)

        return "<0.005" if p == 0 else f"{p:.3g}"

    def draw(ax, row, title):

        obs = [row["top1"], row["top5"], row["top10"]]

        base = [row["base_top1"], row["base_top5"], row["base_top10"]]

        null_mu = [row["null_top1_mean"], row["null_top5_mean"], row["null_top10_mean"]]

        null_lo = [row["null_top1_lo"], row["null_top5_lo"], row["null_top10_lo"]]

        null_hi = [row["null_top1_hi"], row["null_top5_hi"], row["null_top10_hi"]]

        xs = np.arange(3)

        ax.bar(xs - 0.18, obs, width=0.34, color=C["g"], alpha=0.95)

        ax.bar(xs + 0.18, base, width=0.18, color=C["b"], alpha=0.95)

        for i in range(3):

            x_left, x_right = xs[i] - 0.42, xs[i] + 0.42

            ax.fill_between([x_left, x_right], [null_lo[i], null_lo[i]], [null_hi[i], null_hi[i]],

                            color=C["p"], alpha=0.35, linewidth=0)

            ax.hlines(null_mu[i], x_left, x_right, colors=C["p"], linestyles="--", linewidth=1.5)

        for i, v in enumerate(obs):

            ax.text(xs[i]-0.18, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(xs, ["Top-1","Top-5","Top-10"])

        ax.set_ylabel("Retrieval accuracy")

        ax.set_title(title)

        ax.set_ylim(0, min(1.0, max(max(obs), max(null_hi))*1.25 + 1e-6))

        pv = (
            f"p(top1)={fmt_p(row['p_top1'])}\n"
            f"p(top5)={fmt_p(row['p_top5'])}\n"
            f"p(top10)={fmt_p(row['p_top10'])}"
        )

        ax.text(0.98, 0.98, pv, transform=ax.transAxes, ha="right", va="top",

                fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

    row_sample = met[met["task"].eq("sample_pair")].iloc[0].to_dict()

    row_cent = met[met["task"].eq("cultivar_stage_centroid")].iloc[0].to_dict()

    fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.2))

    draw(axs[0], row_sample, "Sample-level (PairID) retrieval")

    draw(axs[1], row_cent, "Cultivar × Stage centroid retrieval")

    obs_proxy = plt.Rectangle((0,0),1,1, fc=C["g"], alpha=0.95)

    base_proxy = plt.Rectangle((0,0),1,1, fc=C["b"], alpha=0.95)

    null_proxy = plt.Line2D([0],[0], color=C["p"], linestyle="--", lw=1.5)

    fig.legend([obs_proxy, base_proxy, null_proxy],

               ["Observed (relative)", "Random baseline", "Null mean; band=95% CI"],

               loc="upper center", ncol=3, frameon=False)

    fig.suptitle("Fig6D | Cross-organ semantic retrieval (relative-weighted)", y=0.98, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_both(fig, outdir / "Fig6D_final.pdf", outdir / "Fig6D_final.svg")

    plt.close(fig)

    fig2, ax2 = plt.subplots(1, 2, figsize=(13.2, 4.8))

    ax2[0].hist(r1, bins=30, color=C["g"], alpha=0.95)

    ax2[0].set_title("Rank distribution | Sample-level")

    ax2[0].set_xlabel(f"Rank of true paired Peel (out of N={len(r1)})")

    ax2[0].set_ylabel("Count")

    ax2[1].hist(r2, bins=20, color=C["p"], alpha=0.90)

    ax2[1].set_title("Rank distribution | Centroid-level")

    ax2[1].set_xlabel(f"Rank of matched Peel centroid (out of N={len(r2)})")

    ax2[1].set_ylabel("Count")

    fig2.tight_layout()

    save_both(fig2, outdir / "Fig6D_rank_distributions_final.pdf", outdir / "Fig6D_rank_distributions_final.svg")

    plt.close(fig2)

def plot_fig7e(root: Path, outdir: Path):

    res = root / "results"

    df = pd.read_csv(res / "Fig6E_robustness_metrics_after_qc_v4.csv")

    scheme_order = ["log1p", "relative", "relative_top20", "relative_top50", "sqrt"]

    df["scheme"] = pd.Categorical(df["scheme"], categories=scheme_order, ordered=True)

    scheme_color = {

        "log1p": C["b"],

        "relative": C["g"],

        "relative_top20": C["r"],

        "relative_top50": C["o"],

        "sqrt": C["p"],

    }

    task_map = {"sample_pair": "Sample pair (PairID)", "cultivar_stage_centroid": "Cultivar × Stage centroid"}

    fig, axs = plt.subplots(2, 1, figsize=(10.8, 9.2), sharex=True)

    for ax, task in zip(axs, ["sample_pair", "cultivar_stage_centroid"]):

        sub = df[df["task"].eq(task)].sort_values("scheme")

        x = np.arange(len(sub)) * 1.00

        y = sub["top10"].to_numpy()

        ylo = sub["top10_lo"].to_numpy()

        yhi = sub["top10_hi"].to_numpy()

        colors = [scheme_color[str(s)] for s in sub["scheme"].astype(str).tolist()]

        ax.bar(x, y, width=0.50, color=colors, alpha=0.95, edgecolor="#333333", linewidth=0.3)

        ax.errorbar(x, y, yerr=[y-ylo, yhi-y], fmt="none", capsize=3, lw=1, color="#222222")

        ax.set_ylabel("Top-10 (95% bootstrap CI)")

        ax.set_title(task_map.get(task, task))

        ax.grid(axis="y", alpha=0.18)

    axs[-1].set_xticks(np.arange(len(scheme_order)) * 1.00, scheme_order, rotation=25, ha="right")

    legend_handles = [plt.Rectangle((0,0),1,1, fc=scheme_color[s], ec="#333333", lw=0.3) for s in scheme_order]

    fig.legend(legend_handles, scheme_order, loc="upper center", ncol=5, frameon=False)

    fig.suptitle("Fig6E | Robustness across weighting schemes after QC filtering", y=0.98, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_both(fig, outdir / "Fig6E_final.pdf", outdir / "Fig6E_final.svg")

    plt.close(fig)

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
    "--root",
    type=str,
    default=str(Path(__file__).resolve().parents[2]),
    )

    ap.add_argument("--out", type=str, default="results/Fig6_final_paletteLocked")

    args = ap.parse_args()

    root = Path(args.root).resolve()

    outdir = Path(args.out)

    if not outdir.is_absolute():

        outdir = (root / outdir).resolve()

    ensure_dir(outdir)

    set_pub_style()

    plot_fig7a(root, outdir)

    plot_fig7b(root, outdir)

    plot_fig7c(root, outdir)

    plot_fig7d(root, outdir)

    plot_fig7e(root, outdir)

    print("[OK] Saved Fig6A–E to:", outdir)

if __name__ == "__main__":

    main()
