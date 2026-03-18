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

GCMS_LEAF = DATA_DIR / "GCMS_leaf.csv"

META_COLS   = ["SampleID", "Cultivar", "Organ", "Stage", "Batch", "Rep"]

STAGE_COL   = "Stage"

STAGE_ORDER = ["S1", "S2", "S3", "S4"]

STAGE_COLORS_LEAF = {

    "S1": "#DCD7EB",

    "S2": "#FCE6CF",

    "S3": "#D5EAD9",

    "S4": "#7DC69B",

}

MODULE_COLORS_LEAF = {

    1: "#F2A1A7",

    2: "#FBDDDD",

    3: "#D8EEFB",

    4: "#DCD7EB",

}

cmap_leaf = LinearSegmentedColormap.from_list(

    "leaf_z", ["#DCD7EB", "#FFFFFF", "#7DC69B"]

)

TOP_N_TRAITS = 80

try:

    from citrus_plot_style import set_citrus_style

    set_citrus_style()

except ImportError:

    pass

print("=== [Fig1C Leaf] 读取 GCMS_leaf.csv ===")

leaf_df = pd.read_csv(GCMS_LEAF)

print("维度:", leaf_df.shape)

print("Stage 分布:")

display(leaf_df[STAGE_COL].value_counts().sort_index())

meta_cols_effective = [c for c in META_COLS if c in leaf_df.columns]

voc_cols_leaf = [c for c in leaf_df.columns if c not in meta_cols_effective]

print(f"[Fig1C Leaf] 识别到 VOC 列数: {len(voc_cols_leaf)}")

stage_series_leaf = leaf_df[STAGE_COL]

leaf_stage_mean = (

    leaf_df[voc_cols_leaf]

    .groupby(stage_series_leaf)

    .mean()

    .reindex(STAGE_ORDER)

)

leaf_stage_mean = leaf_stage_mean.T

print("[Fig1C Leaf] 阶段均值矩阵维度 (VOC × Stage):", leaf_stage_mean.shape)

mean_leaf = leaf_stage_mean.mean(axis=1)

std_leaf  = leaf_stage_mean.std(axis=1, ddof=0)

leaf_z = (leaf_stage_mean.sub(mean_leaf, axis=0)).div(std_leaf.replace(0, np.nan), axis=0)

print("[Fig1C Leaf] Z-score 矩阵维度:", leaf_z.shape)

mean_full_leaf_path = META_DIR / "Fig1C_leaf_stage_mean_raw_full.csv"

z_full_leaf_path    = META_DIR / "Fig1C_leaf_stage_zscore_full.csv"

leaf_stage_mean.to_csv(mean_full_leaf_path)

leaf_z.to_csv(z_full_leaf_path)

print("[Fig1C Leaf] 已保存全量阶段矩阵到:")

print(" -", mean_full_leaf_path)

print(" -", z_full_leaf_path)

print("\n[Fig1C Leaf] 阶段均值矩阵 head():")

display(leaf_stage_mean.head())

print("\n[Fig1C Leaf] Z-score 矩阵 head():")

display(leaf_z.head())

print("\n=== [Fig1C Leaf] 全量聚类与模块划分 ===")

z_full_leaf = leaf_z.dropna(how="all")

print("[Fig1C Leaf] 去除全 NaN 行后 VOC 数:", z_full_leaf.shape[0])

row_link_full_leaf = linkage(z_full_leaf.values, method="ward", metric="euclidean")

modules_full_leaf  = fcluster(row_link_full_leaf, t=4, criterion="maxclust")

z_full_with_module_leaf = z_full_leaf.copy()

z_full_with_module_leaf["module"] = modules_full_leaf

cluster_full_leaf_path = META_DIR / "Fig1C_leaf_stage_zscore_full_with_module.csv"

z_full_with_module_leaf.to_csv(cluster_full_leaf_path)

print("[Fig1C Leaf] 已保存全量 Z-score + module 矩阵到:", cluster_full_leaf_path)

print("[Fig1C Leaf] module 分布（全量）:")

display(pd.Series(modules_full_leaf).value_counts().sort_index())

original_voc_leaf = z_full_leaf.index.to_list()

n_traits_leaf = len(original_voc_leaf)

trait_names_leaf = [f"Leaf Trait {i}" for i in range(1, n_traits_leaf + 1)]

trait_mapping_leaf = dict(zip(original_voc_leaf, trait_names_leaf))

mapping_leaf_df = pd.DataFrame({

    "Trait": [trait_mapping_leaf[v] for v in original_voc_leaf],

    "VOC": original_voc_leaf

})

mapping_leaf_path = META_DIR / "Fig1C_leaf_trait_mapping_full.csv"

mapping_leaf_df.to_csv(mapping_leaf_path, index=False)

print("[Fig1C Leaf] Trait 映射表已保存到:", mapping_leaf_path)

display(mapping_leaf_df.head())

if n_traits_leaf > TOP_N_TRAITS:

    row_var_leaf = z_full_leaf.var(axis=1)

    selected_leaf_vocs = row_var_leaf.sort_values(ascending=False).head(TOP_N_TRAITS).index

else:

    selected_leaf_vocs = z_full_leaf.index

z_leaf_plot = z_full_leaf.loc[selected_leaf_vocs].copy()

z_leaf_plot.index = [trait_mapping_leaf[v] for v in selected_leaf_vocs]

modules_leaf_plot = pd.Series(modules_full_leaf, index=z_full_leaf.index).loc[selected_leaf_vocs]

modules_leaf_plot.index = z_leaf_plot.index

print(f"\n[Fig1C Leaf] 用于绘图的 Trait 数量: {z_leaf_plot.shape[0]}（目标 TOP_N_TRAITS = {TOP_N_TRAITS}）")

z_leaf_plot_with_module = z_leaf_plot.copy()

z_leaf_plot_with_module["module"] = modules_leaf_plot

z_leaf_plot_path = META_DIR / "Fig1C_leaf_stage_zscore_TraitTop80.csv"

z_leaf_plot_with_module.to_csv(z_leaf_plot_path)

print("[Fig1C Leaf] 绘图用 Z-score + module 矩阵已保存到:", z_leaf_plot_path)

row_link_leaf_plot = linkage(z_leaf_plot.values, method="ward", metric="euclidean")

print("\n=== [Fig1C Leaf] 绘制热图（优化布局） ===")

n_plot_leaf = z_leaf_plot.shape[0]

height_leaf = max(12, n_plot_leaf * 0.18)

figsize_leaf = (10.0, height_leaf)

row_colors_leaf = pd.Series(modules_leaf_plot.values, index=z_leaf_plot.index).map(MODULE_COLORS_LEAF)

g_leaf = sns.clustermap(

    z_leaf_plot,

    row_linkage=row_link_leaf_plot,

    col_cluster=False,

    cmap=cmap_leaf,

    center=0,

    vmin=-2, vmax=2,

    row_colors=row_colors_leaf,

    figsize=figsize_leaf,

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

for ax in (g_leaf.ax_row_dendrogram, g_leaf.ax_col_dendrogram):

    for line in ax.get_lines():

        line.set_linewidth(1.5)

g_leaf.ax_heatmap.tick_params(axis="y", labelsize=7)

g_leaf.ax_heatmap.tick_params(axis="x", labelsize=12)

g_leaf.ax_heatmap.set_xlabel("")

g_leaf.ax_heatmap.set_ylabel("")

g_leaf.ax_heatmap.set_title(

    "Leaf VOC developmental modules",

    fontsize=14,

    pad=10

)

g_leaf.fig.subplots_adjust(

    left=0.32,

    right=0.90,

    top=0.96,

    bottom=0.06

)

g_leaf.cax.set_position([1.0, 0.78, 0.03, 0.10])

g_leaf.cax.tick_params(labelsize=9)

plt.show()

base_leaf = OUT_DIR / "Fig1C_leaf_heatmap"

for ext in ["png", "tif"]:

    out_path = base_leaf.with_suffix(f".{ext}")

    g_leaf.fig.savefig(out_path, dpi=600, bbox_inches="tight")

    print("[Fig1C Leaf] 位图已保存:", out_path)

for ext in ["pdf", "svg"]:

    out_path = base_leaf.with_suffix(f".{ext}")

    g_leaf.fig.savefig(out_path, bbox_inches="tight")

    print("[Fig1C Leaf] 矢量图已保存:", out_path)
