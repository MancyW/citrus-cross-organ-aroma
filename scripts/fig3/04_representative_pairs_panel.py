import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

BASE = Path(".").resolve()

DATA = BASE / "data"

META = DATA / "meta"

OUT  = BASE / "output"

META.mkdir(exist_ok=True, parents=True)

OUT.mkdir(exist_ok=True, parents=True)

print("▶ 路径检查：")

print("  DATA =", DATA)

print("  META =", META)

print("  OUT  =", OUT)

mpl.rcParams["font.family"] = "Arial"

mpl.rcParams["pdf.fonttype"] = 42

mpl.rcParams["ps.fonttype"]  = 42

mpl.rcParams["svg.fonttype"] = "none"

STAGE_ORDER = ["S1", "S2", "S3", "S4"]

LEAF_LINE_COLOR = "#7DC69B"

PEEL_LINE_COLOR = "#F2A1A7"

GCMS_PEEL = DATA / "GCMS_peel.csv"

GCMS_LEAF = DATA / "GCMS_leaf.csv"

SEL_FILE  = META / "Fig1D_SelectedPairs_auto.csv"

if not GCMS_PEEL.exists() or not GCMS_LEAF.exists():

    raise FileNotFoundError("找不到 GCMS_peel.csv 或 GCMS_leaf.csv，请检查 data 目录。")

if not SEL_FILE.exists():

    raise FileNotFoundError("找不到 Fig1D_SelectedPairs_auto.csv，请先运行筛选脚本。")

print("\n▶ 读取 GCMS 原始数据和已选 VOC 配对 ...")

df_peel = pd.read_csv(GCMS_PEEL)

df_leaf = pd.read_csv(GCMS_LEAF)

df_sel  = pd.read_csv(SEL_FILE)

print("  Peel 形状:", df_peel.shape)

print("  Leaf 形状:", df_leaf.shape)

print("  选中配对数:", df_sel.shape[0])

META_COLS = ["SampleID", "Cultivar", "Organ", "Stage", "Batch"]

peel_vocs = [c for c in df_peel.columns if c not in META_COLS]

leaf_vocs = [c for c in df_leaf.columns if c not in META_COLS]

if "Organ" in df_peel.columns:

    df_peel = df_peel[df_peel["Organ"].str.lower().str.contains("peel")]

if "Organ" in df_leaf.columns:

    df_leaf = df_leaf[df_leaf["Organ"].str.lower().str.contains("leaf")]

print("\n▶ Step2：为选中配对重新计算 Stage-wise Spearman r ...")

from scipy.stats import spearmanr

corr_records = []

for _, row in df_sel.iterrows():

    pair_id = row["PairID"]

    leaf_voc = row["Leaf_VOC"]

    peel_voc = row["Peel_VOC"]

    if leaf_voc not in leaf_vocs:

        print(f"  [警告] Leaf 中找不到 VOC：{leaf_voc}，跳过该配对。")

        continue

    if peel_voc not in peel_vocs:

        print(f"  [警告] Peel 中找不到 VOC：{peel_voc}，跳过该配对。")

        continue

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

corr_out = META / "Fig1D_StageCorrelation_byPair.csv"

df_corr.to_csv(corr_out, index=False)

print(f"  ✅ 4D-1 相关系数表已保存：{corr_out}")

print("\n▶ Step3：绘制单 panel 4D-1（无 Fisher z 标注） ...")

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

    sub["Stage"] = pd.Categorical(sub["Stage"], categories=STAGE_ORDER, ordered=True)

    sub = sub.sort_values("Stage")

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

ax1.set_title("Fig1D-1 Stage-wise Spearman correlation (Leaf vs Peel VOCs)")

ax1.spines["top"].set_visible(False)

ax1.spines["right"].set_visible(False)

fig1.subplots_adjust(bottom=0.25, top=0.9, right=0.98)

fig1.legend(

    loc="lower center",

    bbox_to_anchor=(0.5, 0.06),

    fontsize=7,

    ncol=2,

    frameon=False

)

out_base1 = OUT / "Fig1D_StageCorrelation_byPair"

for ext in ["pdf", "svg", "png"]:

    fig1.savefig(out_base1.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")

print(f"  ✅ 4D-1 单 panel 图已输出：{out_base1}.[pdf/svg/png]")

plt.close(fig1)

print("\n▶ Step4：为 4D-2 重新计算 Stage 均值 & Z-score ...")

all_vocs = sorted(set(df_sel["Leaf_VOC"]).union(df_sel["Peel_VOC"]))

records_means = []

for organ, df_raw in [("Leaf", df_leaf), ("Peel", df_peel)]:

    for voc in all_vocs:

        if voc not in df_raw.columns:

            print(f"  [警告] {organ} 数据中没有 VOC 列：{voc}，该物质在 4D-2 中将为空。")

            continue

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

        for stage in STAGE_ORDER:

            records_means.append(

                {

                    "Organ": organ,

                    "VOC": voc,

                    "Stage": stage,

                    "MeanAbundance": s_mean.loc[stage],

                    "Zscore": z_vals.loc[stage],

                }

            )

df_means = pd.DataFrame(records_means)

means_out = META / "Fig1D_StageMeans_long.csv"

df_means.to_csv(means_out, index=False)

print(f"  ✅ StageMeans_long 已重算并保存：{means_out}")

print("\n▶ Step5：绘制 4D-2（2×5 panel） ...")

n_pairs = df_sel.shape[0]

n_rows, n_cols = 2, 5

fig2, axes = plt.subplots(

    n_rows, n_cols,

    figsize=(3.2 * n_cols, 3.4 * n_rows),

    sharex=True,

    sharey=True

)

axes = axes.flatten()

x_pos = np.arange(len(STAGE_ORDER))

for ax, (_, sel_row) in zip(axes, df_sel.iterrows()):

    pair_id = sel_row["PairID"]

    leaf_voc = sel_row["Leaf_VOC"]

    peel_voc = sel_row["Peel_VOC"]

    leaf_sub = df_means[(df_means["Organ"] == "Leaf") & (df_means["VOC"] == leaf_voc)].copy()

    peel_sub = df_means[(df_means["Organ"] == "Peel") & (df_means["VOC"] == peel_voc)].copy()

    leaf_sub["Stage"] = pd.Categorical(leaf_sub["Stage"], categories=STAGE_ORDER, ordered=True)

    peel_sub["Stage"] = pd.Categorical(peel_sub["Stage"], categories=STAGE_ORDER, ordered=True)

    leaf_sub.sort_values("Stage", inplace=True)

    peel_sub.sort_values("Stage", inplace=True)

    if leaf_sub.empty or peel_sub.empty:

        ax.text(

            0.5, 0.5,

            "Missing data",

            ha="center", va="center",

            fontsize=8, color="grey",

            transform=ax.transAxes

        )

        ax.set_axis_off()

        continue

    leaf_z = leaf_sub["Zscore"].values

    peel_z = peel_sub["Zscore"].values

    ax.plot(

        x_pos, leaf_z,

        marker="o",

        linestyle="-",

        linewidth=1.5,

        color=LEAF_LINE_COLOR,

        label="Leaf"

    )

    ax.plot(

        x_pos, peel_z,

        marker="s",

        linestyle="-",

        linewidth=1.5,

        color=PEEL_LINE_COLOR,

        label="Peel"

    )

    ax.axhline(0, color="#CCCCCC", linewidth=1, linestyle="--")

    ax.set_xticks(x_pos)

    ax.set_xticklabels(STAGE_ORDER)

    ax.set_title(

        f"{pair_id}: Leaf {leaf_voc}\nPeel {peel_voc}",

        fontsize=8

    )

for k in range(n_pairs, n_rows * n_cols):

    axes[k].set_axis_off()

fig2.supylabel("Z-score of abundance", x=0.01)

fig2.supxlabel("Stage", y=0.05)

handles, labels = axes[0].get_legend_handles_labels()

fig2.legend(

    handles, labels,

    loc="lower center",

    bbox_to_anchor=(0.5, 0.02),

    ncol=2,

    frameon=False

)

fig2.suptitle(

    "Fig1D-2 Stage-wise abundance patterns for selected VOC pairs",

    fontsize=12,

    y=0.98

)

fig2.tight_layout(rect=[0.03, 0.08, 0.97, 0.93])

out_base2 = OUT / "Fig1D_StageAbundance_VOCPairs"

for ext in ["pdf", "svg", "png"]:

    fig2.savefig(out_base2.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")

print(f"  ✅ 4D-2 图像已输出：{out_base2}.[pdf/svg/png]")

plt.close(fig2)

print("\n🎯 Fig1D refine（无 Fisher z 标注版）完成。")
