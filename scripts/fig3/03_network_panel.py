import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.path import Path as MplPath

from matplotlib.patches import PathPatch, Patch

from matplotlib.lines import Line2D

import matplotlib.colors as mcolors

BASE_DIR = Path(".").resolve()

DATA_DIR = BASE_DIR / "data"

META_DIR = DATA_DIR / "meta"

OUT_DIR  = BASE_DIR / "output"

META_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams["font.family"] = "Arial"

mpl.rcParams["pdf.fonttype"] = 42

mpl.rcParams["ps.fonttype"] = 42

mpl.rcParams["svg.fonttype"] = "none"

LINK_ALPHA = 0.7

LINK_LW_MIN = 0.6

LINK_LW_MAX = 3.5

R_ARC   = 1.0

R_NODE  = 0.95

R_LABEL = 1.12

ARC_LW = 8

FAMILY_COLOR_FIXED = {

    "Alkanes/Alkenes":                        "#9BD7F3",

    "Apocarotenoids":                         "#D8EEFB",

    "Benzenoid/Phenylpropanoid-derived volatiles": "#FBDDDD",

    "Fatty acid-derived volatiles":           "#F2A1A7",

    "Monoterpenes":                           "#DCD7EB",

    "N- & heterocycle-containing volatiles":  "#FCE6CF",

    "Sesquiterpenes":                         "#D5EAD9",

}

FAMILY_PALETTE = [

    "#7DC69B", "#9BD7F3", "#D5EAD9", "#D8EEFB",

    "#DCD7EB", "#F2A1A7", "#FBDDDD", "#FCE6CF"

]

PREFERRED_FAMILY_ORDER = [

    "Alkanes/Alkenes",

    "Apocarotenoids",

    "Benzenoid/Phenylpropanoid-derived volatiles",

    "Fatty acid-derived volatiles",

    "Monoterpenes",

    "N- & heterocycle-containing volatiles",

    "Sesquiterpenes",

]

def load_family_annotation():

    class_file = DATA_DIR / "compound_classification.csv"

    voc2family = {}

    if not class_file.exists():

        print("⚠ 未找到 compound_classification.csv，未注释的 VOC 将标记为 Unknown")

        return voc2family

    dfc = pd.read_csv(class_file)

    cols_lower = {c.lower(): c for c in dfc.columns}

    if "compound" not in cols_lower or "family" not in cols_lower:

        print("⚠ compound_classification.csv 中缺少 Compound / Family 列，忽略 family 注释")

        return voc2family

    col_comp = cols_lower["compound"]

    col_fam  = cols_lower["family"]

    dfc["_key"] = dfc[col_comp].astype(str).str.strip().str.lower()

    for _, row in dfc.iterrows():

        key = row["_key"]

        fam = str(row[col_fam]) if pd.notna(row[col_fam]) else "Unknown"

        voc2family[key] = fam

    print(f"✓ Family 注释载入：{len(set(voc2family.values()))} 个 family")

    return voc2family

def ensure_family_columns(edges_df, voc2family):

    df = edges_df.copy()

    if "Leaf_Family" not in df.columns or "Peel_Family" not in df.columns:

        print("▶ 根据 VOC 名称补充 Leaf_Family / Peel_Family 列...")

        df["Leaf_Family"] = (

            df["Leaf_VOC"].astype(str).str.strip().str.lower().map(voc2family).fillna("Unknown")

        )

        df["Peel_Family"] = (

            df["Peel_VOC"].astype(str).str.strip().str.lower().map(voc2family).fillna("Unknown")

        )

    return df

def build_family_color_map(all_families):

    fam2col = {}

    for fam in all_families:

        if fam in FAMILY_COLOR_FIXED:

            fam2col[fam] = FAMILY_COLOR_FIXED[fam]

    palette_idx = 0

    for fam in all_families:

        if fam not in fam2col:

            fam2col[fam] = FAMILY_PALETTE[palette_idx % len(FAMILY_PALETTE)]

            palette_idx += 1

    return fam2col

def order_families(all_fams_set):

    all_fams = list(all_fams_set)

    ordered = []

    for fam in PREFERRED_FAMILY_ORDER:

        if fam in all_fams:

            ordered.append(fam)

    for fam in sorted(all_fams):

        if fam not in ordered:

            ordered.append(fam)

    return ordered

def lighten_or_darken(color, factor=0.3):

    c = np.array(mcolors.to_rgb(color))

    if factor >= 0:

        return tuple(c + (1 - c) * factor)

    else:

        return tuple(c * (1 + factor))

def add_bezier_chord(ax, p1, p2, family_color, r_value, lw):

    x1, y1 = p1

    x2, y2 = p2

    ctrl1 = (x1 * 0.6, y1 * 0.6)

    ctrl2 = (x2 * 0.6, y2 * 0.6)

    verts = [p1, ctrl1, ctrl2, p2]

    codes = [

        MplPath.MOVETO,

        MplPath.CURVE4,

        MplPath.CURVE4,

        MplPath.CURVE4,

    ]

    path = MplPath(verts, codes)

    linestyle = "-" if r_value >= 0 else "--"

    patch = PathPatch(

        path,

        facecolor="none",

        edgecolor=family_color,

        lw=lw,

        alpha=LINK_ALPHA,

        linestyle=linestyle,

    )

    ax.add_patch(patch)

def save_multiformat(fig, base: Path):

    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.2)

    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.2)

    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.2)

    print(

        f"✓ 图形已保存：\n"

        f"  - {base.with_suffix('.pdf')}\n"

        f"  - {base.with_suffix('.svg')}\n"

        f"  - {base.with_suffix('.png')}"

    )

def main():

    voc2family = load_family_annotation()

    peel_file = DATA_DIR / "GCMS_peel.csv"

    df_peel = pd.read_csv(peel_file)

    META_COLS = ["SampleID", "Cultivar", "Organ", "Stage", "Batch"]

    voc_cols = [c for c in df_peel.columns if c not in META_COLS]

    edges_file = META_DIR / "Fig1C_SignificantPairs_networkEdges.csv"

    print("▶ 读取显著 VOC 边：", edges_file)

    edges = pd.read_csv(edges_file)

    required_cols = {"Leaf_VOC", "Peel_VOC", "Spearman_r"}

    if not required_cols.issubset(edges.columns):

        raise ValueError(f"显著边表中缺少必要列：{required_cols - set(edges.columns)}")

    edge_vocs = set(edges["Leaf_VOC"].astype(str)) | set(edges["Peel_VOC"].astype(str))

    extra_vocs = [v for v in sorted(edge_vocs) if v not in voc_cols]

    all_vocs = list(voc_cols) + extra_vocs

    trait_ids = {voc: f"Trait{i}" for i, voc in enumerate(all_vocs, start=1)}

    mapping_records = []

    unknown_records = []

    for voc in all_vocs:

        fam = voc2family.get(voc.strip().lower(), "Unknown")

        mapping_records.append({

            "Trait_ID": trait_ids[voc],

            "VOC_name": voc,

            "Family": fam,

            "In_GCMS_peel": voc in voc_cols,

        })

        if fam == "Unknown":

            unknown_records.append({"Trait_ID": trait_ids[voc], "VOC_name": voc})

    mapping_df = pd.DataFrame(mapping_records)

    map_out = META_DIR / "Fig1C_VOC_ID_mapping.csv"

    mapping_df.to_csv(map_out, index=False)

    print(f"✓ VOC ID 对照表已保存：{map_out}")

    if len(unknown_records) > 0:

        unk_df = pd.DataFrame(unknown_records)

        unk_out = META_DIR / "Fig1C_UnknownFamilyVOCs.csv"

        unk_df.to_csv(unk_out, index=False)

        print(f"⚠ 检测到 {len(unknown_records)} 个 Unknown family，详情已导出：{unk_out}")

        print("   若你确信所有 VOC 都应分类，请检查 compound_classification.csv 名称是否匹配。")

    edges = ensure_family_columns(edges, voc2family)

    edges["Leaf_Family"] = edges["Leaf_Family"].fillna("Unknown")

    edges["Peel_Family"] = edges["Peel_Family"].fillna("Unknown")

    edges = edges[

        (edges["Leaf_Family"] != "Unknown") &

        (edges["Peel_Family"] != "Unknown")

    ].copy()

    print("▶ 聚合到 family × family 层级（用于统计输出）...")

    edges["abs_r"] = edges["Spearman_r"].abs()

    fam_group = (

        edges.groupby(["Leaf_Family", "Peel_Family"], as_index=False)

             .agg(

                 total_weight=("abs_r", "sum"),

                 mean_r=("Spearman_r", "mean"),

                 n_edges=("Spearman_r", "count"),

             )

    )

    fam_out = META_DIR / "Fig1C_FamilyLevelEdges.csv"

    fam_group.to_csv(fam_out, index=False)

    print(f"✓ family-level 边表已保存：{fam_out}")

    leaf_fams = list(fam_group["Leaf_Family"].unique())

    peel_fams = list(fam_group["Peel_Family"].unique())

    all_fams_ordered = order_families(set(leaf_fams) | set(peel_fams))

    family2color = build_family_color_map(all_fams_ordered)

    print(f"  涉及 family 数：{len(all_fams_ordered)}")

    print("▶ 安排 VOC 节点位置（Trait 等距）...")

    leaf_voc_df = (

        edges[["Leaf_VOC", "Leaf_Family"]]

        .drop_duplicates()

        .sort_values(["Leaf_Family", "Leaf_VOC"])

        .reset_index(drop=True)

    )

    peel_voc_df = (

        edges[["Peel_VOC", "Peel_Family"]]

        .drop_duplicates()

        .sort_values(["Peel_Family", "Peel_VOC"])

        .reset_index(drop=True)

    )

    margin = np.deg2rad(3.0)

    theta_leaf_start = np.pi + margin

    theta_leaf_end   = 2 * np.pi - margin

    n_leaf = leaf_voc_df.shape[0]

    d_leaf = (theta_leaf_end - theta_leaf_start) / max(n_leaf, 1)

    leaf_thetas = theta_leaf_start + d_leaf * (np.arange(n_leaf) + 0.5)

    leaf_voc_df["theta"] = leaf_thetas

    theta_peel_start = np.pi - margin

    theta_peel_end   = 0 + margin

    n_peel = peel_voc_df.shape[0]

    d_peel = (theta_peel_end - theta_peel_start) / max(n_peel, 1)

    peel_thetas = theta_peel_start + d_peel * (np.arange(n_peel) + 0.5)

    peel_voc_df["theta"] = peel_thetas

    exp_leaf = 0.1

    exp_peel = 0.1

    arcs_leaf = {}

    for fam, sub in leaf_voc_df.groupby("Leaf_Family", sort=False):

        th_min = sub["theta"].min()

        th_max = sub["theta"].max()

        arcs_leaf[fam] = (

            th_min - exp_leaf * d_leaf,

            th_max + exp_leaf * d_leaf,

            0.5 * (th_min + th_max),

        )

    arcs_peel = {}

    for fam, sub in peel_voc_df.groupby("Peel_Family", sort=False):

        th_min = sub["theta"].min()

        th_max = sub["theta"].max()

        arcs_peel[fam] = (

            th_min - exp_peel * d_peel,

            th_max + exp_peel * d_peel,

            0.5 * (th_min + th_max),

        )

    leaf_node_pos = {}

    for _, row in leaf_voc_df.iterrows():

        voc = row["Leaf_VOC"]

        ang = row["theta"]

        x = R_NODE * np.cos(ang)

        y = R_NODE * np.sin(ang)

        leaf_node_pos[voc] = (x, y, ang)

    peel_node_pos = {}

    for _, row in peel_voc_df.iterrows():

        voc = row["Peel_VOC"]

        ang = row["theta"]

        x = R_NODE * np.cos(ang)

        y = R_NODE * np.sin(ang)

        peel_node_pos[voc] = (x, y, ang)

    node_records = []

    for voc, (x, y, ang) in leaf_node_pos.items():

        node_records.append(

            {"Side": "Leaf", "VOC": voc,

             "Trait_ID": trait_ids.get(voc, "NA"),

             "x": x, "y": y, "theta_rad": ang}

        )

    for voc, (x, y, ang) in peel_node_pos.items():

        node_records.append(

            {"Side": "Peel", "VOC": voc,

             "Trait_ID": trait_ids.get(voc, "NA"),

             "x": x, "y": y, "theta_rad": ang}

        )

    node_df = pd.DataFrame(node_records)

    node_out = META_DIR / "Fig1C_VOC_NodePositions.csv"

    node_df.to_csv(node_out, index=False)

    print(f"✓ VOC 节点坐标已保存：{node_out}")

    print("▶ 绘制 Fig1C 混合 chord 图（Trait 等距，family 彩色弦版）...")

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_aspect("equal")

    ax.axis("off")

    ax.set_xlim(-1.25, 1.25)

    ax.set_ylim(-1.25, 1.25)

    for fam, (theta1, theta2, _) in arcs_leaf.items():

        col = family2color[fam]

        angs = np.linspace(theta1, theta2, 200)

        xs = R_ARC * np.cos(angs)

        ys = R_ARC * np.sin(angs)

        ax.plot(xs, ys, color=col, lw=ARC_LW, solid_capstyle="round")

    for fam, (theta1, theta2, _) in arcs_peel.items():

        col = family2color[fam]

        angs = np.linspace(theta1, theta2, 200)

        xs = R_ARC * np.cos(angs)

        ys = R_ARC * np.sin(angs)

        ax.plot(xs, ys, color=col, lw=ARC_LW, solid_capstyle="round")

    leaf_voc_colors = {}

    peel_voc_colors = {}

    for _, row in leaf_voc_df.iterrows():

        voc = row["Leaf_VOC"]

        fam = row["Leaf_Family"]

        base = family2color[fam]

        leaf_voc_colors[voc] = lighten_or_darken(base, factor=0.35)

    for _, row in peel_voc_df.iterrows():

        voc = row["Peel_VOC"]

        fam = row["Peel_Family"]

        base = family2color[fam]

        peel_voc_colors[voc] = lighten_or_darken(base, factor=-0.2)

    r_abs = edges["abs_r"].values

    r_min = np.nanmin(r_abs)

    r_max = np.nanmax(r_abs)

    def map_lw(r):

        if r_max <= r_min:

            return (LINK_LW_MIN + LINK_LW_MAX) / 2.0

        t = (r - r_min) / (r_max - r_min)

        t = t ** 1.2

        return LINK_LW_MIN + t * (LINK_LW_MAX - LINK_LW_MIN)

    for _, row in edges.iterrows():

        lv = row["Leaf_VOC"]

        pv = row["Peel_VOC"]

        r  = row["Spearman_r"]

        if lv not in leaf_node_pos or pv not in peel_node_pos:

            continue

        leaf_fam = row["Leaf_Family"]

        fam_color = family2color.get(leaf_fam, "#888888")

        p1 = leaf_node_pos[lv][:2]

        p2 = peel_node_pos[pv][:2]

        lw = map_lw(abs(r))

        add_bezier_chord(ax, p1, p2, family_color=fam_color, r_value=r, lw=lw)

    for voc, (x, y, ang) in leaf_node_pos.items():

        ax.scatter(

            x, y,

            s=12,

            facecolor=leaf_voc_colors.get(voc, "#CCCCCC"),

            edgecolor="white",

            linewidth=0.5,

            zorder=3,

        )

    for voc, (x, y, ang) in peel_node_pos.items():

        ax.scatter(

            x, y,

            s=12,

            facecolor=peel_voc_colors.get(voc, "#CCCCCC"),

            edgecolor="white",

            linewidth=0.5,

            zorder=3,

        )

    def add_trait_labels(node_pos):

        for voc, (_, _, ang) in node_pos.items():

            trait_id = trait_ids.get(voc, "")

            if not trait_id:

                continue

            x_t = R_LABEL * np.cos(ang)

            y_t = R_LABEL * np.sin(ang)

            angle_deg = np.rad2deg(ang)

            ax.text(

                x_t,

                y_t,

                trait_id,

                fontsize=5,

                ha="center",

                va="center",

                rotation=angle_deg,

                rotation_mode="anchor",

            )

    add_trait_labels(leaf_node_pos)

    add_trait_labels(peel_node_pos)

    fam_handles = [

        Patch(facecolor=family2color[f], edgecolor="none", label=f)

        for f in all_fams_ordered

    ]

    fam_leg = ax.legend(

        handles=fam_handles,

        title="VOC families",

        loc="upper left",

        bbox_to_anchor=(-0.05, 1.05),

        fontsize=7,

        frameon=False,

    )

    ax.add_artist(fam_leg)

    sign_handles = [

        Line2D([0], [0], color="#555555", lw=2, linestyle="-", label="Positive correlation"),

        Line2D([0], [0], color="#555555", lw=2, linestyle="--", label="Negative correlation"),

    ]

    sign_leg = ax.legend(

        handles=sign_handles,

        loc="upper right",

        bbox_to_anchor=(1.02, 0.96),

        fontsize=7,

        frameon=False,

    )

    ax.add_artist(sign_leg)

    r_levels = np.linspace(r_min, r_max, 3)

    lw_handles = [

        Line2D([0], [0], color="#555555", lw=map_lw(abs(r)), label=f"|r| ≈ {abs(r):.2f}")

        for r in r_levels

    ]

    ax.legend(

        handles=lw_handles,

        title="Correlation strength",

        loc="lower right",

        bbox_to_anchor=(1.05, -0.02),

        fontsize=7,

        frameon=False,

    )

    out_base = OUT_DIR / "Fig1C_MixedChord_FamilyColorChord_Trait_equal"

    save_multiformat(fig, out_base)

    print("▶ Fig1C family 彩色弦版本绘制完成。")

    plt.show()

if __name__ == "__main__":

    main()
