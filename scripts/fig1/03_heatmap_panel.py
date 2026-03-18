import warnings

warnings.filterwarnings(

    "ignore",

    category=UserWarning,

    message="Tight layout not applied"

)

from pathlib import Path

import pandas as pd

import numpy as np

from scipy.cluster.hierarchy import linkage, fcluster

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from IPython.display import display

mpl.rcParams["font.family"] = "Arial"

mpl.rcParams["pdf.fonttype"] = 42

mpl.rcParams["ps.fonttype"] = 42

mpl.rcParams["svg.fonttype"] = "none"

mpl.rcParams["figure.dpi"] = 300

sns.set_theme(style="white")

BASE_DIR = Path(".").resolve()

DATA_DIR = BASE_DIR / "data"

META_DIR = DATA_DIR / "meta"

OUT_DIR  = BASE_DIR / "output"

META_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)

GCMS_PEEL = DATA_DIR / "GCMS_peel.csv"

META_COLS   = ["SampleID", "Cultivar", "Organ", "Stage", "Batch", "Rep"]

STAGE_COL   = "Stage"

STAGE_ORDER = ["S1", "S2", "S3", "S4"]

STAGE_COLORS_PEEL = {

    "S1": "#9BD7F3",

    "S2": "#D8EEFB",

    "S3": "#FBDDDD",

    "S4": "#F2A1A7",

}

MODULE_COLORS_PEEL = {

    1: "#7DC69B",

    2: "#D5EAD9",

    3: "#DCD7EB",

    4: "#CFCFCF",

}

cmap_peel = LinearSegmentedColormap.from_list(

    "peel_z", ["#9BD7F3", "#FFFFFF", "#F2A1A7"]

)

TOP_N_TRAITS = 80

try:

    from citrus_plot_style import set_citrus_style

    set_citrus_style()

except ImportError:

    pass

print("=== [Fig1C Peel] 读取 GCMS_peel.csv ===")

peel_df = pd.read_csv(GCMS_PEEL)

print("维度:", peel_df.shape)

print("Stage 分布:")

display(peel_df[STAGE_COL].value_counts().sort_index())

meta_cols_effective = [c for c in META_COLS if c in peel_df.columns]

voc_cols_peel = [c for c in peel_df.columns if c not in meta_cols_effective]

print(f"[Fig1C Peel] 识别到 VOC 列数: {len(voc_cols_peel)}")

stage_series = peel_df[STAGE_COL]

peel_stage_mean = (

    peel_df[voc_cols_peel]

    .groupby(stage_series)

    .mean()

    .reindex(STAGE_ORDER)

)

peel_stage_mean = peel_stage_mean.T

print("[Fig1C Peel] 阶段均值矩阵维度 (VOC × Stage):", peel_stage_mean.shape)

mean_peel = peel_stage_mean.mean(axis=1)

std_peel  = peel_stage_mean.std(axis=1, ddof=0)

peel_z = (peel_stage_mean.sub(mean_peel, axis=0)).div(std_peel.replace(0, np.nan), axis=0)

print("[Fig1C Peel] Z-score 矩阵维度:", peel_z.shape)

mean_full_path = META_DIR / "Fig1C_peel_stage_mean_raw_full.csv"

z_full_path    = META_DIR / "Fig1C_peel_stage_zscore_full.csv"

peel_stage_mean.to_csv(mean_full_path)

peel_z.to_csv(z_full_path)

print("[Fig1C Peel] 已保存全量阶段矩阵到:")

print(" -", mean_full_path)

print(" -", z_full_path)

print("\n[Fig1C Peel] 阶段均值矩阵 head():")

display(peel_stage_mean.head())

print("\n[Fig1C Peel] Z-score 矩阵 head():")

display(peel_z.head())

print("\n=== [Fig1C Peel] 全量聚类与模块划分 ===")

z_full = peel_z.dropna(how="all")

print("[Fig1C Peel] 去除全 NaN 行后 VOC 数:", z_full.shape[0])

row_link_full = linkage(z_full.values, method="ward", metric="euclidean")

modules_full  = fcluster(row_link_full, t=4, criterion="maxclust")

z_full_with_module = z_full.copy()

z_full_with_module["module"] = modules_full

cluster_full_path = META_DIR / "Fig1C_peel_stage_zscore_full_with_module.csv"

z_full_with_module.to_csv(cluster_full_path)

print("[Fig1C Peel] 已保存全量 Z-score + module 矩阵到:", cluster_full_path)

print("[Fig1C Peel] module 分布（全量）:")

display(pd.Series(modules_full).value_counts().sort_index())

original_voc_names = z_full.index.to_list()

n_traits_full = len(original_voc_names)

trait_names_full = [f"Peel Trait {i}" for i in range(1, n_traits_full + 1)]

trait_mapping = dict(zip(original_voc_names, trait_names_full))

mapping_df = pd.DataFrame({

    "Trait": [trait_mapping[v] for v in original_voc_names],

    "VOC": original_voc_names

})

mapping_path = META_DIR / "Fig1C_peel_trait_mapping_full.csv"

mapping_df.to_csv(mapping_path, index=False)

print("[Fig1C Peel] Trait 映射表已保存到:", mapping_path)

display(mapping_df.head())

if n_traits_full > TOP_N_TRAITS:

    row_var = z_full.var(axis=1)

    selected_vocs = row_var.sort_values(ascending=False).head(TOP_N_TRAITS).index

else:

    selected_vocs = z_full.index

z_plot = z_full.loc[selected_vocs].copy()

z_plot.index = [trait_mapping[v] for v in selected_vocs]

modules_plot = pd.Series(modules_full, index=z_full.index).loc[selected_vocs]

modules_plot.index = z_plot.index

print(f"\n[Fig1C Peel] 用于绘图的 Trait 数量: {z_plot.shape[0]}（目标 TOP_N_TRAITS = {TOP_N_TRAITS}）")

z_plot_with_module = z_plot.copy()

z_plot_with_module["module"] = modules_plot

z_plot_path = META_DIR / "Fig1C_peel_stage_zscore_TraitTop80.csv"

z_plot_with_module.to_csv(z_plot_path)

print("[Fig1C Peel] 绘图用 Z-score + module 矩阵已保存到:", z_plot_path)

row_link_plot = linkage(z_plot.values, method="ward", metric="euclidean")

print("\n=== [Fig1C Peel] 绘制热图（优化布局） ===")

n_plot = z_plot.shape[0]

height = max(12, n_plot * 0.18)

figsize = (10.0, height)

row_colors_peel = pd.Series(modules_plot.values, index=z_plot.index).map(MODULE_COLORS_PEEL)

g_peel = sns.clustermap(

    z_plot,

    row_linkage=row_link_plot,

    col_cluster=False,

    cmap=cmap_peel,

    center=0,

    vmin=-2, vmax=2,

    row_colors=row_colors_peel,

    figsize=figsize,

    xticklabels=True,

    yticklabels=True,

    linewidths=False,

    linecolor="#EAEAEA",

    dendrogram_ratio=(0.15, 0.08),

    cbar_kws={

        "label": "Z-score",

        "ticks": [-2, -1, 0, 1, 2]

    },

)

for ax in (g_peel.ax_row_dendrogram, g_peel.ax_col_dendrogram):

    for line in ax.get_lines():

        line.set_linewidth(1.5)

g_peel.ax_heatmap.tick_params(axis="y", labelsize=7)

g_peel.ax_heatmap.tick_params(axis="x", labelsize=12)

g_peel.ax_heatmap.set_xlabel("")

g_peel.ax_heatmap.set_ylabel("")

g_peel.ax_heatmap.set_title(

    "Peel VOC developmental modules",

    fontsize=14,

    pad=10

)

g_peel.fig.subplots_adjust(

    left=0.32,

    right=0.90,

    top=0.96,

    bottom=0.06

)

g_peel.cax.set_position([1.0, 0.78, 0.03, 0.10])

g_peel.cax.tick_params(labelsize=9)

plt.show()

base = OUT_DIR / "Fig1C_peel_heatmap"

for ext in ["png", "tif"]:

    out_path = base.with_suffix(f".{ext}")

    g_peel.fig.savefig(out_path, dpi=600, bbox_inches="tight")

    print("[Fig1C Peel] 位图已保存:", out_path)

for ext in ["pdf", "svg"]:

    out_path = base.with_suffix(f".{ext}")

    g_peel.fig.savefig(out_path, bbox_inches="tight")

    print("[Fig1C Peel] 矢量图已保存:", out_path)
