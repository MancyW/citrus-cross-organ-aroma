import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

import matplotlib.colors as mcolors

mpl.rcParams["font.family"] = "Arial"

mpl.rcParams["pdf.fonttype"] = 42

mpl.rcParams["ps.fonttype"] = 42

mpl.rcParams["svg.fonttype"] = "none"

PEEL_TOTAL_COLOR = "#9BD7F3"

PEEL_ONLY_COLOR  = "#D8EEFB"

LEAF_TOTAL_COLOR = "#F2A1A7"

LEAF_ONLY_COLOR  = "#FBDDDD"

SHARED_COLOR     = "#FCE6CF"

FAMILY_PALETTE = [

    "#9BD7F3",

    "#D8EEFB",

    "#FBDDDD",

    "#F2A1A7",

    "#DCD7EB",

    "#FCE6CF",

    "#D5EAD9",

    "#7DC69B",

]

PRESENCE_PROP_THRESHOLD = 0.0

META_COLS = ["SampleID", "Cultivar", "Organ", "Stage", "Batch"]

def lighten_color(color_hex, factor=1.25):

    rgb = mcolors.to_rgb(color_hex)

    hsv = mcolors.rgb_to_hsv(rgb)

    hsv[2] = min(1.0, hsv[2] * factor)

    lighter_rgb = mcolors.hsv_to_rgb(hsv)

    return mcolors.to_hex(lighter_rgb)

def smoothstep(t):

    return t * t * (3 - 2 * t)

BASE_DIR = Path(".").resolve()

DATA_DIR = BASE_DIR / "data"

META_DIR = DATA_DIR / "meta"

OUT_DIR  = BASE_DIR / "output"

META_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)

PEEL_FILE  = DATA_DIR / "GCMS_peel.csv"

LEAF_FILE  = DATA_DIR / "GCMS_leaf.csv"

CLASS_FILE = DATA_DIR / "compound_classification.csv"

print("▶ Fig1B 路径设置完成")

print("  DATA_DIR:", DATA_DIR)

print("  META_DIR:", META_DIR)

print("  OUT_DIR :", OUT_DIR)

print("\n▶ 读取 GCMS_peel / GCMS_leaf ...")

df_peel = pd.read_csv(PEEL_FILE)

df_leaf = pd.read_csv(LEAF_FILE)

if "Organ" in df_peel.columns:

    df_peel = df_peel[df_peel["Organ"].str.lower() == "peel"]

if "Organ" in df_leaf.columns:

    df_leaf = df_leaf[df_leaf["Organ"].str.lower() == "leaf"]

voc_cols_peel = [c for c in df_peel.columns if c not in META_COLS]

voc_cols_leaf = [c for c in df_leaf.columns if c not in META_COLS]

all_voc_cols  = sorted(set(voc_cols_peel) | set(voc_cols_leaf))

print(f"  Peel VOC 列数: {len(voc_cols_peel)}")

print(f"  Leaf VOC 列数: {len(voc_cols_leaf)}")

print(f"  VOC 并集列数: {len(all_voc_cols)}")

n_peel = df_peel.shape[0]

n_leaf = df_leaf.shape[0]

print(f"  Peel 样本数: {n_peel}")

print(f"  Leaf 样本数: {n_leaf}")

print("\n▶ 计算 VOC 在 peel / leaf 中的存在情况...")

presence_records = []

for compound in all_voc_cols:

    if compound in df_peel.columns:

        vals_p = df_peel[compound]

        c_p = ((vals_p > 0) & vals_p.notna()).sum()

    else:

        c_p = 0

    if compound in df_leaf.columns:

        vals_l = df_leaf[compound]

        c_l = ((vals_l > 0) & vals_l.notna()).sum()

    else:

        c_l = 0

    prop_p = c_p / n_peel if n_peel > 0 else 0.0

    prop_l = c_l / n_leaf if n_leaf > 0 else 0.0

    present_peel = prop_p > PRESENCE_PROP_THRESHOLD

    present_leaf = prop_l > PRESENCE_PROP_THRESHOLD

    presence_records.append({

        "Compound": compound,

        "Peel_present_count": c_p,

        "Leaf_present_count": c_l,

        "Peel_present_prop": prop_p,

        "Leaf_present_prop": prop_l,

        "Present_in_peel": present_peel,

        "Present_in_leaf": present_leaf

    })

df_presence = pd.DataFrame(presence_records)

mask_peel      = df_presence["Present_in_peel"]

mask_leaf      = df_presence["Present_in_leaf"]

mask_shared    = mask_peel & mask_leaf

mask_peel_only = mask_peel & (~mask_leaf)

mask_leaf_only = (~mask_peel) & mask_leaf

N_peel      = int(mask_peel.sum())

N_leaf      = int(mask_leaf.sum())

N_shared    = int(mask_shared.sum())

N_peel_only = int(mask_peel_only.sum())

N_leaf_only = int(mask_leaf_only.sum())

print("\n▶ 统计 5 类 VOC 数量：")

print("  Peel VOC 总数        :", N_peel)

print("  Leaf VOC 总数        :", N_leaf)

print("  Peel-only VOC 数     :", N_peel_only)

print("  Leaf-only VOC 数     :", N_leaf_only)

print("  Shared (Peel ∩ Leaf) :", N_shared)

presence_outfile = META_DIR / "Fig1B_UpSet_VOC_presence_raw.csv"

df_presence.to_csv(presence_outfile, index=False)

print("  ▶ 已保存 presence 表:", presence_outfile)

def assign_category(row):

    if row["Present_in_peel"] and row["Present_in_leaf"]:

        return "Shared"

    elif row["Present_in_peel"]:

        return "Peel-only"

    elif row["Present_in_leaf"]:

        return "Leaf-only"

    else:

        return "None"

df_presence["Category"] = df_presence.apply(assign_category, axis=1)

df_presence_sub = df_presence[df_presence["Category"].isin(["Peel-only", "Shared", "Leaf-only"])].copy()

print("\n▶ 读取 compound_classification 并计算 family 组成...")

df_class = pd.read_csv(CLASS_FILE)

if "Compound" not in df_class.columns or "Family" not in df_class.columns:

    raise ValueError("compound_classification.csv 中必须包含 'Compound' 和 'Family' 两列。")

df_class_sub = df_class[["Compound", "Family"]].dropna(subset=["Family"]).copy()

df_merged = pd.merge(

    df_presence_sub,

    df_class_sub,

    on="Compound",

    how="inner"

)

print("  合并后 VOC 数（有 Family 注释）：", df_merged.shape[0])

family_order = sorted(df_merged["Family"].unique().tolist())

n_family = len(family_order)

print("  Family 种类数:", n_family)

print("  Family 列表:", family_order)

family_color_map = {}

for i, fam in enumerate(family_order):

    family_color_map[fam] = FAMILY_PALETTE[i % len(FAMILY_PALETTE)]

cat_order = ["Peel-only", "Shared", "Leaf-only"]

records = []

for cat in cat_order:

    sub = df_merged[df_merged["Category"] == cat]

    total = sub.shape[0]

    if total == 0:

        for fam in family_order:

            records.append({"Category": cat, "Family": fam, "Count": 0, "Percent": 0.0})

        continue

    counts = sub.groupby("Family")["Compound"].nunique()

    for fam in family_order:

        c = counts.get(fam, 0)

        pct = c / total * 100

        records.append({"Category": cat, "Family": fam, "Count": c, "Percent": pct})

df_family_long = pd.DataFrame(records)

family_outfile = META_DIR / "Fig1B_FamilyComposition_long.csv"

df_family_long.to_csv(family_outfile, index=False)

print("  ▶ 已保存 family 组成长表:", family_outfile)

pivot_pct = df_family_long.pivot(index="Family", columns="Category", values="Percent").fillna(0)

pivot_pct = pivot_pct.loc[family_order, cat_order]

print("\n▶ 绘制 Fig1B UpSet + family flow ...")

fig = plt.figure(figsize=(6.5, 6.8), dpi=300)

gs = GridSpec(

    3, 1,

    height_ratios=[2.7, 1.1, 3.0],

    hspace=0.35,

    figure=fig

)

ax_bar = fig.add_subplot(gs[0, 0])

x_peel   = 0

x_shared = 1

x_leaf   = 2

bar_width_total  = 0.7

bar_width_nested = 0.35

ax_bar.bar(x_peel,   N_peel,   width=bar_width_total,

           color=PEEL_TOTAL_COLOR, edgecolor="none")

ax_bar.bar(x_shared, N_shared, width=bar_width_total,

           color=SHARED_COLOR,     edgecolor="none")

ax_bar.bar(x_leaf,   N_leaf,   width=bar_width_total,

           color=LEAF_TOTAL_COLOR, edgecolor="none")

if N_peel_only > 0:

    ax_bar.bar(x_peel, N_peel_only,

               width=bar_width_nested,

               color=PEEL_ONLY_COLOR, edgecolor="none")

if N_leaf_only > 0:

    ax_bar.bar(x_leaf, N_leaf_only,

               width=bar_width_nested,

               color=LEAF_ONLY_COLOR, edgecolor="none")

max_size = max(N_peel, N_leaf, N_shared, 1)

ax_bar.text(x_peel,   N_peel   + max_size * 0.03, str(N_peel),

            ha="center", va="bottom", fontsize=9)

ax_bar.text(x_shared, N_shared + max_size * 0.03, str(N_shared),

            ha="center", va="bottom", fontsize=9)

ax_bar.text(x_leaf,   N_leaf   + max_size * 0.03, str(N_leaf),

            ha="center", va="bottom", fontsize=9)

if N_peel_only > 0:

    ax_bar.text(x_peel, N_peel_only / 2, str(N_peel_only),

                ha="center", va="center", fontsize=8, color="black")

if N_leaf_only > 0:

    ax_bar.text(x_leaf, N_leaf_only / 2, str(N_leaf_only),

                ha="center", va="center", fontsize=8, color="black")

ax_bar.set_xticks([x_peel, x_shared, x_leaf])

ax_bar.set_xticklabels(["Peel", "Shared", "Leaf"], fontsize=9)

ax_bar.set_ylabel("Number of VOCs", fontsize=10)

ax_bar.set_xlim(-0.8, 2.8)

ax_bar.spines["top"].set_visible(False)

ax_bar.spines["right"].set_visible(False)

ax_bar.spines["left"].set_linewidth(1.0)

ax_bar.spines["bottom"].set_linewidth(1.0)

handles = [

    plt.Rectangle((0, 0), 1, 1, color=PEEL_TOTAL_COLOR),

    plt.Rectangle((0, 0), 1, 1, color=LEAF_TOTAL_COLOR),

    plt.Rectangle((0, 0), 1, 1, color=PEEL_ONLY_COLOR),

    plt.Rectangle((0, 0), 1, 1, color=LEAF_ONLY_COLOR),

    plt.Rectangle((0, 0), 1, 1, color=SHARED_COLOR),

]

labels = [

    "Peel VOC (total)",

    "Leaf VOC (total)",

    "Peel-only",

    "Leaf-only",

    "Shared"

]

ax_bar.legend(

    handles, labels,

    fontsize=8, title_fontsize=9,

    title="VOC groups",

    loc="center left",

    bbox_to_anchor=(1.02, 1.0),

    frameon=False

)

ax_mat = fig.add_subplot(gs[1, 0])

x_int = np.array([x_peel, x_shared, x_leaf])

intersections = pd.DataFrame({

    "Category": ["Peel_only", "Shared", "Leaf_only"],

    "Size":     [N_peel_only, N_shared, N_leaf_only],

    "Label":    ["Peel-only", "Shared", "Leaf-only"],

    "x":        x_int

})

set_names = ["Peel", "Leaf"]

y_map = {"Peel": 1, "Leaf": 0}

ax_mat.set_xlim(-0.8, 2.8)

ax_mat.set_ylim(-0.7, 1.5)

for xi in x_int:

    for s in set_names:

        ax_mat.scatter(xi, y_map[s], s=28, color="#d0d0d0", zorder=1)

for _, row in intersections.iterrows():

    xi   = row["x"]

    cat  = row["Category"]

    size = row["Size"]

    if cat == "Peel_only":

        present = ["Peel"]

    elif cat == "Leaf_only":

        present = ["Leaf"]

    else:

        present = ["Peel", "Leaf"]

    for s in present:

        c = PEEL_TOTAL_COLOR if s == "Peel" else LEAF_TOTAL_COLOR

        ax_mat.scatter(xi, y_map[s], s=42, color=c, zorder=2)

    if cat == "Shared":

        ax_mat.plot(

            [xi, xi],

            [y_map["Leaf"], y_map["Peel"]],

            "-", color=SHARED_COLOR,

            linewidth=1.3, zorder=1.5

        )

    ax_mat.text(

        xi, 1.25,

        str(int(size)),

        ha="center", va="bottom",

        fontsize=8

    )

ax_mat.set_xticks(x_int)

ax_mat.set_xticklabels(intersections["Label"].values, fontsize=9)

ax_mat.set_yticks([y_map[s] for s in set_names])

ax_mat.set_yticklabels(set_names, fontsize=9)

for spine in ax_mat.spines.values():

    spine.set_visible(False)

ax_mat.tick_params(axis="x", length=0)

ax_mat.tick_params(axis="y", length=0)

ax_flow = fig.add_subplot(gs[2, 0])

x = np.array([x_peel, x_shared, x_leaf])

x_labels = ["Peel-only", "Shared", "Leaf-only"]

width = 0.7

bottom_transition = np.zeros(len(x))

for fam in family_order:

    vals = pivot_pct.loc[fam, cat_order].values

    color_base = family_color_map[fam]

    color_flow = lighten_color(color_base, factor=1.20)

    for i in range(len(x) - 1):

        x0 = x[i]   + width / 2.0

        x1 = x[i+1] - width / 2.0

        y0_bottom = bottom_transition[i]

        y0_top    = bottom_transition[i] + vals[i]

        y1_bottom = bottom_transition[i+1]

        y1_top    = bottom_transition[i+1] + vals[i+1]

        t  = np.linspace(0, 1, 40)

        st = smoothstep(t)

        x_curve = x0 + (x1 - x0) * t

        y_bottom_curve = y0_bottom + (y1_bottom - y0_bottom) * st

        y_top_curve    = y0_top    + (y1_top    - y0_top)    * st

        poly_x = np.concatenate([x_curve, x_curve[::-1]])

        poly_y = np.concatenate([y_bottom_curve, y_top_curve[::-1]])

        ax_flow.fill(

            poly_x, poly_y,

            color=color_flow,

            alpha=0.40,

            linewidth=0

        )

    bottom_transition += vals

bottom = np.zeros(len(x))

for fam in family_order:

    vals = pivot_pct.loc[fam, cat_order].values

    color_bar = family_color_map[fam]

    ax_flow.bar(

        x, vals,

        width=width,

        bottom=bottom,

        color=color_bar,

        edgecolor="white",

        linewidth=0.8

    )

    bottom += vals

ax_flow.set_xticks(x)

ax_flow.set_xticklabels(x_labels, fontsize=9)

ax_flow.set_ylabel("Family composition (%)", fontsize=10)

ax_flow.set_ylim(0, 100)

ax_flow.set_xlim(-0.8, 2.8)

ax_flow.spines["top"].set_visible(False)

ax_flow.spines["right"].set_visible(False)

ax_flow.spines["left"].set_linewidth(1.0)

ax_flow.spines["bottom"].set_linewidth(1.0)

ax_flow.tick_params(axis="x", length=0)

ax_flow.tick_params(axis="y", length=3)

handles_fam = [

    plt.Rectangle((0, 0), 1, 1, color=family_color_map[fam])

    for fam in family_order

]

ax_flow.legend(

    handles_fam, family_order,

    title="VOC families",

    fontsize=8, title_fontsize=9,

    loc="center left",

    bbox_to_anchor=(1.02, 0.5),

    frameon=False

)

plt.tight_layout(rect=[0, 0, 1.0, 1.0])

fig_base = OUT_DIR / "Fig1B_UpSet_FamilyFlow"

for ext in ["pdf", "svg", "png"]:

    out_file = f"{fig_base}.{ext}"

    if ext == "png":

        fig.savefig(out_file, dpi=600, bbox_inches="tight")

    else:

        fig.savefig(out_file, bbox_inches="tight")

    print("  ▶ 已保存图像:", out_file)

plt.show()

print("\n✅ Fig1B UpSet + family flow 图绘制完成。")
