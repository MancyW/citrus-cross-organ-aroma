import warnings

warnings.filterwarnings("ignore")

import os

from pathlib import Path

import pandas as pd

import numpy as np

from scipy.stats import kruskal

from sklearn.preprocessing import StandardScaler

from sklearn.cross_decomposition import PLSRegression

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib as mpl

from matplotlib.ticker import FormatStrFormatter

mpl.rcParams["font.family"] = "Arial"

mpl.rcParams["pdf.fonttype"] = 42

mpl.rcParams["ps.fonttype"] = 42

mpl.rcParams["svg.fonttype"] = "none"

BASE_DIR = Path(".").resolve()

DATA_DIR = BASE_DIR / "data"

META_DIR = DATA_DIR / "meta"

OUT_DIR  = BASE_DIR / "output"

META_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)

PEEL_FILE = DATA_DIR / "GCMS_peel.csv"

print(">>> [Path] 读取文件:", PEEL_FILE)

df_raw = pd.read_csv(PEEL_FILE)

meta_cols = ["SampleID", "Cultivar", "Organ", "Stage", "Batch", "Rep"]

voc_cols = [c for c in df_raw.columns if c not in meta_cols]

stage_order = ["S1", "S2", "S3", "S4"]

df_raw["Stage"] = pd.Categorical(df_raw["Stage"], categories=stage_order, ordered=True)

print(f">>> 样本数: {df_raw.shape[0]}, VOC 数量: {len(voc_cols)}")

df = df_raw.copy()

df[voc_cols] = df[voc_cols].fillna(0)

print(">>> Step0 Done: NaN→0")

valid_vocs = [v for v in voc_cols if (df[v] == 0).mean() < 0.7]

print(f">>> 稀疏过滤后 VOC 数量: {len(valid_vocs)}")

df_log = df.copy()

df_log[valid_vocs] = np.log2(df_log[valid_vocs] + 1e-6)

cultivars = df["Cultivar"].unique()

step1_counts = {}

for voc in valid_vocs:

    cnt = 0

    for cv in cultivars:

        sub = df_log[df_log["Cultivar"] == cv]

        groups = [sub.loc[sub.Stage == s, voc].values for s in stage_order]

        if any(len(g) == 0 for g in groups):

            continue

        try:

            _, p = kruskal(*groups)

            if p < 0.05:

                cnt += 1

        except:

            continue

    step1_counts[voc] = cnt

step1_sig = pd.DataFrame(step1_counts.items(), columns=["VOC", "Count"]).query("Count>=5")

step1_set = set(step1_sig["VOC"])

print(f">>> Step1 ≥5 个品种显著的 VOC: {len(step1_set)}")

df_log = df_log.replace([np.inf, -np.inf], np.nan)

valid_vocs_step2 = [v for v in valid_vocs if df_log[v].notna().all()]

print(f">>> Step2 可用于 PLS 的 VOC: {len(valid_vocs_step2)}")

X = df_log[valid_vocs_step2].values

y = df["Stage"].cat.codes.values

X_scaled = StandardScaler().fit_transform(X)

pls = PLSRegression(n_components=2).fit(X_scaled, y)

T, W, Q = pls.x_scores_, pls.x_weights_, pls.y_loadings_

vip_scores = []

p, h = W.shape

SStotal = np.sum((T**2) * (Q**2), axis=0)

for i in range(p):

    vip = np.sqrt(p * np.sum((W[i]**2) * SStotal) / np.sum(SStotal))

    vip_scores.append(vip)

vip_sig = pd.DataFrame({"VOC": valid_vocs_step2, "VIP": vip_scores}).query("VIP>1")

step2_set = set(vip_sig["VOC"])

print(f">>> Step2 VIP>1 数量: {len(step2_set)}")

rf = RandomForestClassifier(n_estimators=800, random_state=0).fit(X_scaled, y)

rf_df = pd.DataFrame({"VOC": valid_vocs_step2, "Imp": rf.feature_importances_})

rf_df = rf_df.sort_values("Imp", ascending=False)

top_k = int(len(valid_vocs_step2) * 0.12)

rf_top = rf_df.head(top_k)

step3_set = set(rf_top["VOC"])

print(f">>> Step3 RF Top12%: {len(step3_set)}")

freq_dict = {}

for voc in valid_vocs_step2:

    cnt = 0

    for cv in cultivars:

        sub = df_log[df_log["Cultivar"] == cv]

        groups = [sub.loc[sub.Stage == s, voc].values for s in stage_order]

        if any(len(g) == 0 for g in groups):

            continue

        try:

            _, p = kruskal(*groups)

            if p < 0.05:

                cnt += 1

        except:

            continue

    freq_dict[voc] = cnt

freq_sig = pd.DataFrame(freq_dict.items(), columns=["VOC", "Freq"]).query("Freq>=10")

step4_set = set(freq_sig["VOC"])

print(f">>> Step4 ≥10 cultivars: {len(step4_set)}")

final_vocs = list(step1_set & step2_set & step3_set & step4_set)

print(">>> Final VOC:", final_vocs)

stage_colors = {

    "S1": "#9BD7F3", "S2": "#D8EEFB", "S3": "#FBDDDD", "S4": "#F2A1A7"

}

print("\n>>> 开始绘图（矢量可编辑 + PNG/TIF 位图）")

if len(final_vocs) == 0:

    print("!!! 无关键 VOC，无法绘图")

else:

    df_z = df.copy()

    df_z[final_vocs] = (df[final_vocs] - df[final_vocs].mean()) / df[final_vocs].std()

    plot_df = df_z[["Stage"] + final_vocs].melt(

        id_vars="Stage", var_name="VOC", value_name="Z"

    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes = axes.flatten()

    for ax, voc in zip(axes, final_vocs):

        sub = plot_df[plot_df["VOC"] == voc]

        sns.swarmplot(

            data=sub, x="Stage", y="Z",

            order=stage_order,

            palette=stage_colors,

            size=6, alpha=1.0, linewidth=0,

            ax=ax

        )

        means = sub.groupby("Stage")["Z"].mean().reindex(stage_order)

        sds   = sub.groupby("Stage")["Z"].std().reindex(stage_order)

        x     = np.arange(len(stage_order))

        ax.fill_between(

            x, means - sds, means + sds,

            color="#CCCCCC", alpha=0.20

        )

        ax.plot(

            x, means.values,

            color="black", linewidth=2.2,

            marker="o", markersize=8,

            markerfacecolor="white",

            markeredgecolor="black"

        )

        ax.spines["top"].set_visible(False)

        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(1.2)

        ax.spines["bottom"].set_linewidth(1.2)

        ax.set_xticks(x)

        ax.set_xticklabels(stage_order, fontsize=12)

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        ax.tick_params(axis="y", labelsize=12)

        ax.set_ylim(-1.5, 4)

        ax.set_yticks(np.linspace(-1, 3, 5))

        ax.set_title(voc, fontsize=16, pad=10)

        ax.set_xlabel("Stage", fontsize=14)

        ax.set_ylabel("Z-score", fontsize=14)

    for j in range(len(final_vocs), 4):

        axes[j].axis("off")

    plt.tight_layout()

    for ext in ["png", "tif", "svg", "pdf"]:

        out_path = OUT_DIR / f"Fig1E_keyVOC_trends_swarm_editable.{ext}"

        plt.savefig(out_path, dpi=600)

        print(">>> 输出文件:", out_path)

print("\n>>> Done: Fig1E（可编辑矢量 + 位图）已全部生成！")
