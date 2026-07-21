import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm


# ============================================================
# Path settings
# ============================================================

BASE = Path(".").resolve()

DATA = BASE / "data"
META = DATA / "meta"
OUT = BASE / "output"

META.mkdir(exist_ok=True, parents=True)
OUT.mkdir(exist_ok=True, parents=True)

print("▶ 路径检查：")
print("  DATA =", DATA)
print("  META =", META)
print("  OUT  =", OUT)


# ============================================================
# Plot settings
# ============================================================

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

STAGE_ORDER = ["S1", "S2", "S3", "S4"]

LEAF_LINE_COLOR = "#7DC69B"
PEEL_LINE_COLOR = "#F2A1A7"

# If True, Fig. 3e subpanels will show only P1, P2, ...
# If False, subpanels will show PairID + VOC names.
COMPACT_PANEL_TITLE = True

# If True, add a figure-level suptitle for debugging.
# For manuscript figure assembly, keep it False.
SHOW_DEBUG_TITLES = False


# ============================================================
# Input files
# ============================================================

GCMS_PEEL = DATA / "GCMS_peel.csv"
GCMS_LEAF = DATA / "GCMS_leaf.csv"

# Compatible with several historical filenames.
PAIR_FILE_CANDIDATES = [
    META / "Fig3d_SelectedPairs_auto.csv",
    META / "Fig4D_SelectedPairs_auto.csv",
    META / "Fig1D_SelectedPairs_auto.csv",
    DATA / "Fig3d_SelectedPairs_auto.csv",
    DATA / "Fig4D_SelectedPairs_auto.csv",
    DATA / "Fig1D_SelectedPairs_auto.csv",
]

SEL_FILE = None
for candidate in PAIR_FILE_CANDIDATES:
    if candidate.exists():
        SEL_FILE = candidate
        break

if not GCMS_PEEL.exists() or not GCMS_LEAF.exists():
    raise FileNotFoundError(
        "找不到 GCMS_peel.csv 或 GCMS_leaf.csv。请确认这两个文件位于 data 目录。"
    )

if SEL_FILE is None:
    raise FileNotFoundError(
        "找不到 SelectedPairs 文件。请将 Fig3d_SelectedPairs_auto.csv、"
        "Fig4D_SelectedPairs_auto.csv 或 Fig1D_SelectedPairs_auto.csv 放入 data/meta 或 data 目录。"
    )


# ============================================================
# Helper functions
# ============================================================

def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg FDR correction.
    Returns q values in the original order.
    """
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return q

    p_valid = p[valid]
    n = len(p_valid)
    order = np.argsort(p_valid)
    ranked = p_valid[order]

    q_ranked = ranked * n / (np.arange(n) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.minimum(q_ranked, 1.0)

    q_valid = np.empty_like(q_ranked)
    q_valid[order] = q_ranked

    q[valid] = q_valid
    return q


def fisher_r_to_z_test(r1, n1, r2, n2):
    """
    Two-sided Fisher r-to-z test for difference between two independent correlations.
    This is used here as an approximate comparison of stage-stratified correlations.
    """
    if any(pd.isna(v) for v in [r1, n1, r2, n2]):
        return np.nan, np.nan

    if n1 <= 3 or n2 <= 3:
        return np.nan, np.nan

    # Prevent infinite arctanh when r is exactly ±1.
    r1_clip = np.clip(r1, -0.999999, 0.999999)
    r2_clip = np.clip(r2, -0.999999, 0.999999)

    z1 = np.arctanh(r1_clip)
    z2 = np.arctanh(r2_clip)

    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_stat = (z1 - z2) / se
    p_val = 2 * (1 - norm.cdf(abs(z_stat)))

    return z_stat, p_val


def safe_stage_categorical(df, stage_col="Stage"):
    df = df.copy()
    df[stage_col] = pd.Categorical(
        df[stage_col],
        categories=STAGE_ORDER,
        ordered=True
    )
    return df.sort_values(stage_col)


# ============================================================
# Read data
# ============================================================

print("\n▶ 读取 GCMS 原始数据和已选 VOC 配对 ...")

df_peel = pd.read_csv(GCMS_PEEL)
df_leaf = pd.read_csv(GCMS_LEAF)
df_sel = pd.read_csv(SEL_FILE)

print("  Peel 形状:", df_peel.shape)
print("  Leaf 形状:", df_leaf.shape)
print("  Selected pairs 文件:", SEL_FILE)
print("  选中配对数:", df_sel.shape[0])

# Required columns for selected pair file.
required_pair_cols = {"PairID", "Leaf_VOC", "Peel_VOC"}
missing_pair_cols = required_pair_cols - set(df_sel.columns)
if missing_pair_cols:
    raise ValueError(f"SelectedPairs 文件缺少列: {missing_pair_cols}")

META_COLS = ["SampleID", "Cultivar", "Organ", "Stage", "Batch"]

if "Organ" in df_peel.columns:
    df_peel = df_peel[df_peel["Organ"].astype(str).str.lower().str.contains("peel")]

if "Organ" in df_leaf.columns:
    df_leaf = df_leaf[df_leaf["Organ"].astype(str).str.lower().str.contains("leaf")]

peel_vocs = [c for c in df_peel.columns if c not in META_COLS]
leaf_vocs = [c for c in df_leaf.columns if c not in META_COLS]

# Keep only selected pairs with valid VOC names.
valid_rows = []
for _, row in df_sel.iterrows():
    leaf_voc = row["Leaf_VOC"]
    peel_voc = row["Peel_VOC"]

    if leaf_voc not in leaf_vocs:
        print(f"  [警告] Leaf 中找不到 VOC：{leaf_voc}，该配对将跳过。")
        continue

    if peel_voc not in peel_vocs:
        print(f"  [警告] Peel 中找不到 VOC：{peel_voc}，该配对将跳过。")
        continue

    valid_rows.append(row)

df_sel = pd.DataFrame(valid_rows).reset_index(drop=True)
print("  有效配对数:", df_sel.shape[0])

if df_sel.empty:
    raise ValueError("没有有效 VOC 配对，请检查 SelectedPairs 文件中的 VOC 名称。")


# ============================================================
# Step 2: Stage-wise Spearman correlations for Fig. 3d
# ============================================================

print("\n▶ Step2：重新计算 Fig. 3d stage-wise Spearman correlations ...")

corr_records = []

for _, row in df_sel.iterrows():
    pair_id = row["PairID"]
    leaf_voc = row["Leaf_VOC"]
    peel_voc = row["Peel_VOC"]

    for stage in STAGE_ORDER:
        tmp_leaf = (
            df_leaf[df_leaf["Stage"] == stage]
            .groupby("Cultivar")[leaf_voc]
            .mean()
        )

        tmp_peel = (
            df_peel[df_peel["Stage"] == stage]
            .groupby("Cultivar")[peel_voc]
            .mean()
        )

        common = tmp_leaf.index.intersection(tmp_peel.index)

        x = tmp_leaf.loc[common].values
        y = tmp_peel.loc[common].values

        if len(common) < 3:
            r_val = np.nan
            p_val = np.nan
        else:
            r_val, p_val = spearmanr(x, y)

        corr_records.append(
            {
                "PairID": pair_id,
                "Leaf_VOC": leaf_voc,
                "Peel_VOC": peel_voc,
                "Stage": stage,
                "n_cultivars": len(common),
                "Spearman_r": r_val,
                "Spearman_p": p_val,
            }
        )

df_corr = pd.DataFrame(corr_records)

# FDR correction for all stage-wise correlations.
df_corr["Spearman_q_FDR"] = benjamini_hochberg(df_corr["Spearman_p"].values)

corr_out = META / "Fig3d_StageCorrelation_byPair.csv"
df_corr.to_csv(corr_out, index=False)
print(f"  ✅ Fig. 3d stage-wise correlation table saved: {corr_out}")


# ============================================================
# Step 2b: Fisher r-to-z tests for stage-to-stage differences
# ============================================================

print("\n▶ Step2b：计算 Fisher r-to-z stage-to-stage correlation difference tests ...")

fisher_records = []

stage_pairs = []
for i, s1 in enumerate(STAGE_ORDER):
    for s2 in STAGE_ORDER[i + 1:]:
        stage_pairs.append((s1, s2))

for _, sel_row in df_sel.iterrows():
    pair_id = sel_row["PairID"]
    leaf_voc = sel_row["Leaf_VOC"]
    peel_voc = sel_row["Peel_VOC"]

    sub = df_corr[df_corr["PairID"] == pair_id].copy()

    for s1, s2 in stage_pairs:
        row1 = sub[sub["Stage"] == s1]
        row2 = sub[sub["Stage"] == s2]

        if row1.empty or row2.empty:
            z_stat, p_val = np.nan, np.nan
            r1 = r2 = n1 = n2 = np.nan
        else:
            r1 = float(row1["Spearman_r"].iloc[0])
            r2 = float(row2["Spearman_r"].iloc[0])
            n1 = int(row1["n_cultivars"].iloc[0])
            n2 = int(row2["n_cultivars"].iloc[0])
            z_stat, p_val = fisher_r_to_z_test(r1, n1, r2, n2)

        fisher_records.append(
            {
                "PairID": pair_id,
                "Leaf_VOC": leaf_voc,
                "Peel_VOC": peel_voc,
                "Stage_1": s1,
                "Stage_2": s2,
                "Spearman_r_stage_1": r1,
                "n_stage_1": n1,
                "Spearman_r_stage_2": r2,
                "n_stage_2": n2,
                "Fisher_z_statistic": z_stat,
                "Fisher_p_value": p_val,
            }
        )

df_fisher = pd.DataFrame(fisher_records)
df_fisher["Fisher_q_FDR"] = benjamini_hochberg(df_fisher["Fisher_p_value"].values)

fisher_out = META / "Fig3d_Fisher_r_to_z_tests.csv"
df_fisher.to_csv(fisher_out, index=False)
print(f"  ✅ Fisher r-to-z tests saved: {fisher_out}")


# ============================================================
# Step 3: Plot Fig. 3d
# ============================================================

print("\n▶ Step3：绘制 Fig. 3d stage-wise Spearman correlation panel ...")

fig1, ax1 = plt.subplots(figsize=(7.0, 6.0))

x_pos = np.arange(len(STAGE_ORDER))
pair_colors = plt.cm.tab10(np.linspace(0, 1, df_sel.shape[0]))

for color, (_, sel_row) in zip(pair_colors, df_sel.iterrows()):
    pair_id = sel_row["PairID"]
    leaf_voc = sel_row["Leaf_VOC"]
    peel_voc = sel_row["Peel_VOC"]

    sub = df_corr[df_corr["PairID"] == pair_id].copy()
    if sub.empty:
        continue

    sub = safe_stage_categorical(sub)

    y = sub["Spearman_r"].values

    ax1.plot(
        x_pos,
        y,
        marker="o",
        linestyle="-",
        linewidth=1.8,
        color=color,
        label=f"{pair_id}: {leaf_voc} ↔ {peel_voc}"
    )

ax1.axhline(0, color="lightgray", linestyle="--", linewidth=1)

ax1.set_xlim(-0.3, len(STAGE_ORDER) - 0.7)
ax1.set_ylim(-1.05, 1.05)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(STAGE_ORDER)

ax1.set_ylabel("Spearman r (leaf–peel per stage)")
ax1.set_xlabel("Stage")

if SHOW_DEBUG_TITLES:
    ax1.set_title("Fig. 3d Stage-wise Spearman correlation")

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig1.subplots_adjust(bottom=0.25, top=0.92, right=0.98)

fig1.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 0.06),
    fontsize=7,
    ncol=2,
    frameon=False
)

out_base1 = OUT / "Fig3d_StageCorrelation_byPair"

for ext in ["pdf", "svg", "png"]:
    fig1.savefig(out_base1.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")

print(f"  ✅ Fig. 3d saved: {out_base1}.[pdf/svg/png]")

plt.close(fig1)


# ============================================================
# Step 4: Stage means, Z-scores, SD and SEM for Fig. 3e
# ============================================================

print("\n▶ Step4：重新计算 Fig. 3e stage means, Z-scores, SD and SEM ...")

all_vocs = sorted(set(df_sel["Leaf_VOC"]).union(df_sel["Peel_VOC"]))

records_means = []
records_stats = []

for organ, df_raw in [("Leaf", df_leaf), ("Peel", df_peel)]:
    for voc in all_vocs:
        if voc not in df_raw.columns:
            print(f"  [警告] {organ} 数据中没有 VOC 列：{voc}，该物质在 Fig. 3e 中将为空。")
            continue

        # ----------------------------------------------------
        # 1) Keep the original plotted trajectory definition:
        #    stage mean abundance across samples -> Z-score across S1-S4.
        # ----------------------------------------------------
        s_mean = (
            df_raw
            .groupby("Stage")[voc]
            .mean()
            .reindex(STAGE_ORDER)
        )

        mu = s_mean.mean()
        sigma = s_mean.std(ddof=0)

        if sigma == 0 or pd.isna(sigma):
            z_vals = (s_mean - mu) * 0
        else:
            z_vals = (s_mean - mu) / sigma

        # ----------------------------------------------------
        # 2) Variability across cultivars:
        #    cultivar × stage mean abundance -> same Z-score scale
        #    using the mu/sigma from the plotted trajectory.
        # ----------------------------------------------------
        cult_stage = (
            df_raw
            .groupby(["Cultivar", "Stage"], as_index=False)[voc]
            .mean()
            .rename(columns={voc: "CultivarStageMeanAbundance"})
        )

        if sigma == 0 or pd.isna(sigma):
            cult_stage["CultivarStageZscore"] = 0.0
        else:
            cult_stage["CultivarStageZscore"] = (
                cult_stage["CultivarStageMeanAbundance"] - mu
            ) / sigma

        stage_stats = (
            cult_stage
            .groupby("Stage")["CultivarStageZscore"]
            .agg(
                mean_Z_by_cultivar="mean",
                sd_Z_across_cultivars=lambda x: x.std(ddof=1),
                n_cultivars="count"
            )
            .reindex(STAGE_ORDER)
            .reset_index()
        )

        stage_stats["sem_Z_across_cultivars"] = (
            stage_stats["sd_Z_across_cultivars"] / np.sqrt(stage_stats["n_cultivars"])
        )

        for stage in STAGE_ORDER:
            stat_row = stage_stats[stage_stats["Stage"] == stage].iloc[0]

            records_means.append(
                {
                    "Organ": organ,
                    "VOC": voc,
                    "Stage": stage,
                    "MeanAbundance": s_mean.loc[stage],
                    "Zscore": z_vals.loc[stage],
                    "mean_Z_by_cultivar": stat_row["mean_Z_by_cultivar"],
                    "sd_Z_across_cultivars": stat_row["sd_Z_across_cultivars"],
                    "sem_Z_across_cultivars": stat_row["sem_Z_across_cultivars"],
                    "n_cultivars": int(stat_row["n_cultivars"]),
                }
            )

            records_stats.append(
                {
                    "Organ": organ,
                    "VOC": voc,
                    "Stage": stage,
                    "mean_abundance_plotted": s_mean.loc[stage],
                    "mean_Zscore_plotted": z_vals.loc[stage],
                    "mean_Z_by_cultivar": stat_row["mean_Z_by_cultivar"],
                    "SD_Zscore_across_cultivars": stat_row["sd_Z_across_cultivars"],
                    "SEM_Zscore_across_cultivars": stat_row["sem_Z_across_cultivars"],
                    "n_cultivars": int(stat_row["n_cultivars"]),
                }
            )

df_means = pd.DataFrame(records_means)
df_stats = pd.DataFrame(records_stats)

means_out = META / "Fig3e_StageMeans_long.csv"
stats_out = META / "Fig3e_trajectory_mean_SD_SEM.csv"

df_means.to_csv(means_out, index=False)
df_stats.to_csv(stats_out, index=False)

print(f"  ✅ Fig. 3e StageMeans_long saved: {means_out}")
print(f"  ✅ Fig. 3e trajectory mean/SD/SEM table saved: {stats_out}")


# ============================================================
# Step 5: Plot Fig. 3e with SEM error bars
# ============================================================

print("\n▶ Step5：绘制 Fig. 3e trajectory panel with SEM error bars ...")

n_pairs = df_sel.shape[0]
n_rows, n_cols = 2, 5

fig2, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(3.2 * n_cols, 3.4 * n_rows),
    sharex=True,
    sharey=True
)

axes = axes.flatten()
x_pos = np.arange(len(STAGE_ORDER))

# For dynamic y-limits.
all_y_for_ylim = []

for ax, (_, sel_row) in zip(axes, df_sel.iterrows()):
    pair_id = sel_row["PairID"]
    leaf_voc = sel_row["Leaf_VOC"]
    peel_voc = sel_row["Peel_VOC"]

    leaf_sub = df_means[
        (df_means["Organ"] == "Leaf") &
        (df_means["VOC"] == leaf_voc)
    ].copy()

    peel_sub = df_means[
        (df_means["Organ"] == "Peel") &
        (df_means["VOC"] == peel_voc)
    ].copy()

    leaf_sub = safe_stage_categorical(leaf_sub)
    peel_sub = safe_stage_categorical(peel_sub)

    if leaf_sub.empty or peel_sub.empty:
        ax.text(
            0.5,
            0.5,
            "Missing data",
            ha="center",
            va="center",
            fontsize=8,
            color="grey",
            transform=ax.transAxes
        )
        ax.set_axis_off()
        continue

    leaf_z = leaf_sub["Zscore"].values
    peel_z = peel_sub["Zscore"].values

    leaf_sem = leaf_sub["sem_Z_across_cultivars"].values
    peel_sem = peel_sub["sem_Z_across_cultivars"].values

    all_y_for_ylim.extend((leaf_z - leaf_sem).tolist())
    all_y_for_ylim.extend((leaf_z + leaf_sem).tolist())
    all_y_for_ylim.extend((peel_z - peel_sem).tolist())
    all_y_for_ylim.extend((peel_z + peel_sem).tolist())

    ax.errorbar(
        x_pos,
        leaf_z,
        yerr=leaf_sem,
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=4,
        capsize=2,
        elinewidth=0.8,
        capthick=0.8,
        color=LEAF_LINE_COLOR,
        label="Leaf"
    )

    ax.errorbar(
        x_pos,
        peel_z,
        yerr=peel_sem,
        marker="s",
        linestyle="-",
        linewidth=1.5,
        markersize=4,
        capsize=2,
        elinewidth=0.8,
        capthick=0.8,
        color=PEEL_LINE_COLOR,
        label="Peel"
    )

    ax.axhline(0, color="#CCCCCC", linewidth=1, linestyle="--")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(STAGE_ORDER)

    if COMPACT_PANEL_TITLE:
        ax.set_title(f"{pair_id}", fontsize=9)
    else:
        ax.set_title(
            f"{pair_id}: Leaf {leaf_voc}\nPeel {peel_voc}",
            fontsize=8
        )

for k in range(n_pairs, n_rows * n_cols):
    axes[k].set_axis_off()

# Dynamic shared y-axis limit.
all_y_for_ylim = np.asarray(all_y_for_ylim, dtype=float)
all_y_for_ylim = all_y_for_ylim[~np.isnan(all_y_for_ylim)]

if len(all_y_for_ylim) > 0:
    ylim_abs = np.nanmax(np.abs(all_y_for_ylim))
    ylim_abs = max(1.25, min(2.75, ylim_abs + 0.20))
else:
    ylim_abs = 1.25

for ax in axes[:n_pairs]:
    ax.set_ylim(-ylim_abs, ylim_abs)

fig2.supylabel("Z-score of abundance", x=0.01)
fig2.supxlabel("Stage", y=0.05)

handles, labels = axes[0].get_legend_handles_labels()

fig2.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=2,
    frameon=False
)

if SHOW_DEBUG_TITLES:
    fig2.suptitle(
        "Fig. 3e Stage-wise abundance patterns for selected VOC pairs",
        fontsize=12,
        y=0.98
    )
    fig2.tight_layout(rect=[0.03, 0.08, 0.97, 0.93])
else:
    fig2.tight_layout(rect=[0.03, 0.08, 0.97, 0.97])

out_base2 = OUT / "Fig3e_StageAbundance_VOCPairs_with_SEM"

for ext in ["pdf", "svg", "png"]:
    fig2.savefig(out_base2.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")

print(f"  ✅ Fig. 3e with SEM error bars saved: {out_base2}.[pdf/svg/png]")

plt.close(fig2)


# ============================================================
# Step 6: Export additional Source Data workbook
# ============================================================

print("\n▶ Step6：导出 Fig. 3d/e additional source data workbook ...")

source_excel = META / "Fig3d_e_additional_statistics_for_SourceData.xlsx"

with pd.ExcelWriter(source_excel, engine="openpyxl") as writer:
    df_sel.to_excel(writer, sheet_name="Selected pairs", index=False)
    df_corr.to_excel(writer, sheet_name="Fig3d stage correlations", index=False)
    df_fisher.to_excel(writer, sheet_name="Fig3d Fisher r-to-z", index=False)
    df_means.to_excel(writer, sheet_name="Fig3e plotted data SEM", index=False)
    df_stats.to_excel(writer, sheet_name="Fig3e mean SD SEM", index=False)

print(f"  ✅ Source data workbook saved: {source_excel}")

print("\n🎯 Fig. 3d/e plotting with SEM error bars completed.")