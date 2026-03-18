import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from scipy.spatial.distance import pdist, squareform

from scipy.stats import gaussian_kde

import numpy.linalg as la

mpl.rcParams["font.family"]   = "Arial"

mpl.rcParams["pdf.fonttype"]  = 42

mpl.rcParams["ps.fonttype"]   = 42

mpl.rcParams["svg.fonttype"]  = "none"

mpl.rcParams["figure.dpi"]    = 300

BASE_DIR = Path(".").resolve()

DATA_DIR = BASE_DIR / "data"

META_DIR = DATA_DIR / "meta"

OUT_DIR  = BASE_DIR / "output"

META_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)

PEEL_FILE = DATA_DIR / "GCMS_peel.csv"

LEAF_FILE = DATA_DIR / "GCMS_leaf.csv"

print("▶ Fig1A: 路径检查")

print("  DATA_DIR :", DATA_DIR)

print("  META_DIR :", META_DIR)

print("  OUT_DIR  :", OUT_DIR)

print("  PEEL_FILE:", PEEL_FILE.exists())

print("  LEAF_FILE:", LEAF_FILE.exists())

print("\n▶ Fig1A: 读取原始数据 ...")

df_peel = pd.read_csv(PEEL_FILE)

df_leaf = pd.read_csv(LEAF_FILE)

meta_cols = ["SampleID", "Cultivar", "Stage", "Organ"]

for col in meta_cols:

    if col not in df_peel.columns or col not in df_leaf.columns:

        raise ValueError(f"缺少元信息列: {col}")

voc_cols_peel = [c for c in df_peel.columns if c not in meta_cols]

voc_cols_leaf = [c for c in df_leaf.columns if c not in meta_cols]

all_voc_cols = sorted(set(voc_cols_peel) | set(voc_cols_leaf))

def align_voc_columns(df, meta_cols, all_voc_cols):

    df = df.copy()

    for c in all_voc_cols:

        if c not in df.columns:

            df[c] = 0.0

    return df[meta_cols + all_voc_cols]

df_peel = align_voc_columns(df_peel, meta_cols, all_voc_cols)

df_leaf = align_voc_columns(df_leaf, meta_cols, all_voc_cols)

df_all = pd.concat([df_peel, df_leaf], axis=0, ignore_index=True)

print(f"  peel 样本数: {df_peel.shape[0]}")

print(f"  leaf 样本数: {df_leaf.shape[0]}")

print(f"  合并后样本数: {df_all.shape[0]}")

print(f"  VOC 变量数: {len(all_voc_cols)}")

print("\n▶ Fig1A: 标准化元信息列 ...")

df_all["Organ"] = (

    df_all["Organ"]

    .astype(str)

    .str.strip()

    .str.lower()

    .replace({

        "peel": "peel",

        "fruit peel": "peel",

        "rind": "peel",

        "leaf": "leaf",

        "leaves": "leaf",

    })

)

df_all["Stage"] = df_all["Stage"].astype(str).str.strip()

df_all["Stage"] = df_all["Stage"].str.replace("^([0-9])$", r"S\1", regex=True)

stage_order = ["S1", "S2", "S3", "S4"]

df_all["Stage"] = pd.Categorical(df_all["Stage"],

                                 categories=stage_order,

                                 ordered=True)

df_all = df_all[df_all["Organ"].isin(["peel", "leaf"])].reset_index(drop=True)

print("\n▶ Fig1A: VOC 矩阵 log10 + autoscale 处理（用于 PCA） ...")

nan_mask = df_all[all_voc_cols].isna()

n_nan_cells = int(nan_mask.sum().sum())

n_nan_samples = int(nan_mask.any(axis=1).sum())

print(f"  发现 NaN 单元格数量: {n_nan_cells}")

print(f"  含 NaN 的样本数量:   {n_nan_samples}")

if n_nan_cells > 0:

    nan_report = df_all[meta_cols].copy()

    nan_report["n_nan_vocs"] = nan_mask.sum(axis=1)

    nan_report_out = META_DIR / "Fig1A_missing_values_report.csv"

    nan_report.to_csv(nan_report_out, index=False)

    print(f"  已输出缺失值报告: {nan_report_out}")

    print("  处理策略：将 VOC 中 NaN 视为 0（未检测）。")

df_all[all_voc_cols] = df_all[all_voc_cols].fillna(0.0)

X_raw = df_all[all_voc_cols].to_numpy(dtype=float)

neg_mask = X_raw < 0

n_neg_cells = int(neg_mask.sum())

n_neg_samples = int(neg_mask.any(axis=1).sum())

print(f"  发现负值单元格数量: {n_neg_cells}")

print(f"  含负值的样本数量:   {n_neg_samples}")

if n_neg_cells > 0:

    neg_report = df_all[meta_cols].copy()

    neg_report["n_negative_vocs"] = neg_mask.sum(axis=1)

    neg_report_out = META_DIR / "Fig1A_negative_values_report.csv"

    neg_report.to_csv(neg_report_out, index=False)

    print(f"  已输出负值报告: {neg_report_out}")

    print("  处理策略：负值截断为 0。")

    X_raw[neg_mask] = 0.0

df_all[all_voc_cols] = X_raw

X_log = np.log10(X_raw + 1.0)

if np.isnan(X_log).any() or np.isinf(X_log).any():

    raise ValueError("log10 之后存在 NaN/Inf，请检查数据。")

scaler = StandardScaler(with_mean=True, with_std=True)

X_scaled = scaler.fit_transform(X_log)

if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():

    raise ValueError("StandardScaler 之后存在 NaN/Inf，请检查数据。")

scaled_df = pd.concat(

    [

        df_all[meta_cols].reset_index(drop=True),

        pd.DataFrame(X_scaled, columns=all_voc_cols),

    ],

    axis=1

)

scaled_out = META_DIR / "Fig1A_PCA_input_log10_autoscale.csv"

scaled_df.to_csv(scaled_out, index=False)

print(f"  已输出 PCA 输入矩阵: {scaled_out}")

print("\n▶ Fig1A-PCA: 执行 PCA ...")

pca = PCA(n_components=4, random_state=0)

scores = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_ * 100

pc1_var = explained[0]

pc2_var = explained[1]

print(f"  PC1 解释方差: {pc1_var:.2f}%")

print(f"  PC2 解释方差: {pc2_var:.2f}%")

scores_df = df_all[meta_cols].copy()

scores_df["PC1"] = scores[:, 0]

scores_df["PC2"] = scores[:, 1]

scores_df["PC3"] = scores[:, 2]

scores_df["PC4"] = scores[:, 3]

scores_out = META_DIR / "Fig1A_PCA_scores_all_samples.csv"

scores_df.to_csv(scores_out, index=False)

print(f"  已输出 PCA 得分表: {scores_out}")

centroids_cs = (

    scores_df

    .groupby(["Cultivar", "Stage", "Organ"], dropna=False)[["PC1", "PC2"]]

    .mean()

    .reset_index()

)

centroids_cs_out = META_DIR / "Fig1A_PCA_centroids_by_Cultivar_Stage_Organ.csv"

centroids_cs.to_csv(centroids_cs_out, index=False)

print(f"  已输出 PCA 质心表: {centroids_cs_out}")

print("\n▶ Fig1A: 设置配色与点型 ...")

peel_stage_colors = {

    "S1": "#9BD7F3",

    "S2": "#D8EEFB",

    "S3": "#FBDDDD",

    "S4": "#F2A1A7",

}

leaf_stage_colors = {

    "S1": "#DCD7EB",

    "S2": "#FCE6CF",

    "S3": "#D5EAD9",

    "S4": "#7DC69B",

}

color_map = {}

for st in stage_order:

    color_map[("leaf", st)] = leaf_stage_colors.get(st, "#999999")

    color_map[("peel", st)] = peel_stage_colors.get(st, "#666666")

organ_marker = {

    "leaf": "^",

    "peel": "o",

}

organ_order = ["leaf", "peel"]

print("\n▶ Fig1A-PCA: 绘制 PCA 点云 ...")

fig, ax = plt.subplots(figsize=(6.0, 6.0))

handles_dict = {}

for organ in organ_order:

    for st in stage_order:

        mask = (scores_df["Organ"] == organ) & (scores_df["Stage"] == st)

        if mask.sum() == 0:

            continue

        sub = scores_df.loc[mask]

        c = color_map[(organ, st)]

        m = organ_marker[organ]

        label = f"{organ.capitalize()} {st}"

        sc = ax.scatter(

            sub["PC1"], sub["PC2"],

            s=26,

            marker=m,

            color=c,

            edgecolor="black",

            linewidth=0.4,

            alpha=0.7,

            zorder=2,

        )

        if label not in handles_dict:

            handles_dict[label] = sc

handles = list(handles_dict.values())

labels  = list(handles_dict.keys())

ax.set_xlabel(f"PC1 ({pc1_var:.1f}% var. expl.)", fontsize=12)

ax.set_ylabel(f"PC2 ({pc2_var:.1f}% var. expl.)", fontsize=12)

ax.axhline(0, color="#dddddd", linewidth=0.8, zorder=1)

ax.axvline(0, color="#dddddd", linewidth=0.8, zorder=1)

ax.grid(False)

for spine in ["top", "right"]:

    ax.spines[spine].set_visible(False)

legend = ax.legend(

    handles, labels,

    title="Organ × Stage",

    bbox_to_anchor=(1.02, 1.0),

    loc="upper left",

    borderaxespad=0.5,

    fontsize=8,

    title_fontsize=9,

    frameon=False,

)

ax.set_title("Fig1A  Peel + Leaf VOCs PCA", fontsize=12, pad=10)

fig.tight_layout()

out_base_pca = OUT_DIR / "Fig1A_PCA_peel_leaf_cloud"

fig.savefig(out_base_pca.with_suffix(".pdf"), format="pdf", bbox_inches="tight")

fig.savefig(out_base_pca.with_suffix(".svg"), format="svg", bbox_inches="tight")

print(f"  已输出: {out_base_pca.with_suffix('.pdf')}")

print(f"  已输出: {out_base_pca.with_suffix('.svg')}")

plt.show()

print("\n▶ Fig1A-PCoA: 基于 Bray–Curtis 的 PCoA ...")

X_bc_raw = df_all[all_voc_cols].to_numpy(dtype=float)

X_bc_raw[X_bc_raw < 0] = 0.0

X_bc = np.log10(X_bc_raw + 1.0)

bc_condensed = pdist(X_bc, metric="braycurtis")

bc_dm = squareform(bc_condensed)

n = bc_dm.shape[0]

print(f"  距离矩阵大小: {bc_dm.shape}")

print("  执行经典 MDS ...")

I = np.eye(n)

one = np.ones((n, n)) / n

J = I - one

D2 = bc_dm ** 2

B = -0.5 * J.dot(D2).dot(J)

eigvals, eigvecs = la.eigh(B)

idx = np.argsort(eigvals)[::-1]

eigvals = eigvals[idx]

eigvecs = eigvecs[:, idx]

positive = eigvals > 0

eigvals_pos = eigvals[positive]

eigvecs_pos = eigvecs[:, positive]

coords_all = eigvecs_pos * np.sqrt(eigvals_pos)

if coords_all.shape[1] < 2:

    raise ValueError("正特征值少于 2 个，PCoA 无法得到二维坐标。")

coords_2d = coords_all[:, :2]

pcoa_var1 = float(eigvals_pos[0] / eigvals_pos.sum() * 100)

pcoa_var2 = float(eigvals_pos[1] / eigvals_pos.sum() * 100)

print(f"  PCoA1 解释方差: {pcoa_var1:.2f}%")

print(f"  PCoA2 解释方差: {pcoa_var2:.2f}%")

coords_df = pd.concat(

    [

        df_all[meta_cols].reset_index(drop=True),

        pd.DataFrame(coords_2d, columns=["PCoA1", "PCoA2"]),

    ],

    axis=1

)

coords_out = META_DIR / "Fig1A_PCoA_BrayCurtis_coords.csv"

coords_df.to_csv(coords_out, index=False)

print(f"  已输出 PCoA 坐标表: {coords_out}")

permanova_input = pd.concat(

    [

        df_all[meta_cols].reset_index(drop=True),

        pd.DataFrame(X_bc, columns=all_voc_cols),

    ],

    axis=1

)

permanova_input_out = META_DIR / "Fig1A_PERMANOVA_input_log10.csv"

permanova_input.to_csv(permanova_input_out, index=False)

print(f"  已输出多因子 PERMANOVA 输入矩阵: {permanova_input_out}")

print("\n▶ Fig1A-PCoA: 计算一元 PERMANOVA (Organ, Stage) ...")

def permanova_one_factor(D, groups, n_perm=999, random_state=0):

    rng = np.random.default_rng(random_state)

    D2 = D ** 2

    n = D2.shape[0]

    uniq, grp_idx = np.unique(groups, return_inverse=True)

    k = len(uniq)

    iu = np.triu_indices(n, 1)

    SST = D2[iu].sum() / n

    def ss_within(g_idx):

        SSW = 0.0

        for g in range(k):

            mask = (g_idx == g)

            ng = mask.sum()

            if ng <= 1:

                continue

            sub = D2[np.ix_(mask, mask)]

            iu2 = np.triu_indices(ng, 1)

            SSW += sub[iu2].sum() / ng

        return SSW

    SSW = ss_within(grp_idx)

    SSB = SST - SSW

    df_between = k - 1

    df_within  = n - k

    MSB = SSB / df_between

    MSW = SSW / df_within

    F_obs = MSB / MSW

    R2 = SSB / SST

    count = 0

    for _ in range(n_perm):

        perm = rng.permutation(grp_idx)

        SSW_p = ss_within(perm)

        SSB_p = SST - SSW_p

        MSB_p = SSB_p / df_between

        MSW_p = SSW_p / df_within

        F_p = MSB_p / MSW_p

        if F_p >= F_obs - 1e-12:

            count += 1

    p_val = (count + 1) / (n_perm + 1)

    return R2, p_val, F_obs

R2_organ, p_organ, F_organ = permanova_one_factor(

    bc_dm,

    df_all["Organ"].to_numpy(),

    n_perm=999,

    random_state=123,

)

print(f"  PERMANOVA (Organ): R² = {R2_organ:.3f}, F = {F_organ:.3f}, p = {p_organ:.4f}")

R2_stage, p_stage, F_stage = permanova_one_factor(

    bc_dm,

    df_all["Stage"].astype(str).to_numpy(),

    n_perm=999,

    random_state=456,

)

print(f"  PERMANOVA (Stage): R² = {R2_stage:.3f}, F = {F_stage:.3f}, p = {p_stage:.4f}")

permanova_summary = pd.DataFrame({

    "Factor": ["Organ", "Stage"],

    "R2":     [R2_organ, R2_stage],

    "F":      [F_organ, F_stage],

    "p":      [p_organ, p_stage],

    "n_perm": [999, 999],

})

permanova_out = META_DIR / "Fig1A_PERMANOVA_onefactor_Organ_Stage.csv"

permanova_summary.to_csv(permanova_out, index=False)

print(f"  已输出一元 PERMANOVA 结果表: {permanova_out}")

print("\n▶ Fig1A-PCoA: 绘制 PCoA 点云 + 边缘密度 ...")

x_min = coords_df["PCoA1"].min()

x_max = coords_df["PCoA1"].max()

y_min = coords_df["PCoA2"].min()

y_max = coords_df["PCoA2"].max()

x_pad = (x_max - x_min) * 0.15

y_pad = (y_max - y_min) * 0.28

x_lim = (x_min - x_pad, x_max + x_pad)

y_lim = (y_min - y_pad, y_max + y_pad)

fig = plt.figure(figsize=(6.5, 6.5))

ax_scatter = fig.add_axes([0.14, 0.14, 0.60, 0.60])

ax_top = fig.add_axes([0.14, 0.76, 0.60, 0.12], sharex=ax_scatter)

ax_right = fig.add_axes([0.76, 0.14, 0.12, 0.60], sharey=ax_scatter)

handles_dict = {}

for organ in organ_order:

    for st in stage_order:

        mask = (coords_df["Organ"] == organ) & (coords_df["Stage"] == st)

        if mask.sum() == 0:

            continue

        sub = coords_df.loc[mask]

        c = color_map[(organ, st)]

        m = organ_marker[organ]

        label = f"{organ.capitalize()} {st}"

        sc = ax_scatter.scatter(

            sub["PCoA1"], sub["PCoA2"],

            s=26,

            marker=m,

            color=c,

            edgecolor="black",

            linewidth=0.4,

            alpha=0.7,

            zorder=2,

        )

        if label not in handles_dict:

            handles_dict[label] = sc

handles = list(handles_dict.values())

labels  = list(handles_dict.keys())

ax_scatter.set_xlim(*x_lim)

ax_scatter.set_ylim(*y_lim)

ax_scatter.set_xlabel(

    f"PCoA1 ({pcoa_var1:.1f}%)",

    fontsize=12,

    labelpad=10,

)

ax_scatter.set_ylabel(

    f"PCoA2 ({pcoa_var2:.1f}%)",

    fontsize=12,

    labelpad=10,

)

ax_scatter.axhline(0, color="#dddddd", linewidth=0.8, zorder=1)

ax_scatter.axvline(0, color="#dddddd", linewidth=0.8, zorder=1)

ax_scatter.grid(False)

for spine in ["top", "right", "left", "bottom"]:

    ax_scatter.spines[spine].set_visible(True)

legend = ax_scatter.legend(

    handles, labels,

    title="Organ × Stage",

    loc="lower left",

    fontsize=8,

    title_fontsize=9,

    frameon=False,

)

txt = (f"PERMANOVA (Bray–Curtis)\n"

       f"Organ: R² = {R2_organ:.3f}, p = {p_organ:.3f}\n"

       f"Stage: R² = {R2_stage:.3f}, p = {p_stage:.3f}")

x_range = x_lim[1] - x_lim[0]

y_range = y_lim[1] - y_lim[0]

ax_scatter.text(

    x_lim[1] - 0.03 * x_range,

    y_lim[1] - 0.06 * y_range,

    txt,

    ha="right",

    va="top",

    fontsize=9,

)

x_grid = np.linspace(*x_lim, 200)

top_max = 0.0

for organ in organ_order:

    for st in stage_order:

        mask = (coords_df["Organ"] == organ) & (coords_df["Stage"] == st)

        x_vals = coords_df.loc[mask, "PCoA1"].values

        if x_vals.size < 5:

            continue

        kde = gaussian_kde(x_vals)

        y_vals = kde(x_grid)

        top_max = max(top_max, y_vals.max())

        c = color_map[(organ, st)]

        ax_top.fill_between(

            x_grid, 0, y_vals,

            color=c,

            alpha=0.25,

            linewidth=0,

        )

        ax_top.plot(x_grid, y_vals, color=c, alpha=0.9, linewidth=1.3)

if top_max > 0:

    ax_top.set_ylim(0, top_max * 1.05)

ax_top.set_xlim(*x_lim)

y_grid = np.linspace(*y_lim, 200)

right_max = 0.0

for organ in organ_order:

    for st in stage_order:

        mask = (coords_df["Organ"] == organ) & (coords_df["Stage"] == st)

        y_vals = coords_df.loc[mask, "PCoA2"].values

        if y_vals.size < 5:

            continue

        kde = gaussian_kde(y_vals)

        x_vals = kde(y_grid)

        right_max = max(right_max, x_vals.max())

        c = color_map[(organ, st)]

        ax_right.fill_betweenx(

            y_grid, 0, x_vals,

            color=c,

            alpha=0.25,

            linewidth=0,

        )

        ax_right.plot(x_vals, y_grid, color=c, alpha=0.9, linewidth=1.3)

if right_max > 0:

    ax_right.set_xlim(0, right_max * 1.05)

ax_right.set_ylim(*y_lim)

for ax in [ax_top, ax_right]:

    ax.tick_params(left=False, bottom=False,

                   labelleft=False, labelbottom=False)

    ax.patch.set_alpha(0.0)

    for spine in ["top", "right", "left", "bottom"]:

        ax.spines[spine].set_visible(False)

out_base_pcoa = OUT_DIR / "Fig1A_PCoA_peel_leaf_BrayCurtis_cloud_marginal"

fig.savefig(out_base_pcoa.with_suffix(".pdf"),

            format="pdf", bbox_inches="tight")

fig.savefig(out_base_pcoa.with_suffix(".svg"),

            format="svg", bbox_inches="tight")

print(f"  已输出: {out_base_pcoa.with_suffix('.pdf')}")

print(f"  已输出: {out_base_pcoa.with_suffix('.svg')}")

plt.show()

print("\n✔ Fig1A 脚本执行完成。")

print("  - PCA 输入矩阵:   data/meta/Fig1A_PCA_input_log10_autoscale.csv")

print("  - PCA 得分表:     data/meta/Fig1A_PCA_scores_all_samples.csv")

print("  - PCA 质心表:     data/meta/Fig1A_PCA_centroids_by_Cultivar_Stage_Organ.csv")

print("  - PCoA 坐标表:    data/meta/Fig1A_PCoA_BrayCurtis_coords.csv")

print("  - PERMANOVA 输入: data/meta/Fig1A_PERMANOVA_input_log10.csv")

print("  - 一元 PERMANOVA: data/meta/Fig1A_PERMANOVA_onefactor_Organ_Stage.csv")

print("  - PCA 图:         output/Fig1A_PCA_peel_leaf_cloud.(pdf/svg)")

print("  - PCoA+边缘图:    output/Fig1A_PCoA_peel_leaf_BrayCurtis_cloud_marginal.(pdf/svg)")

import numpy as np

import pandas as pd

import numpy.linalg as la

print("\n▶ Fig1A-PERMANOVA: 多因子 Organ + Stage + Cultivar (Python 版本) ...")

G = B.copy()

n = G.shape[0]

G = 0.5 * (G + G.T)

def build_design(meta_df, factors):

    n_samples = meta_df.shape[0]

    cols_full = [np.ones((n_samples, 1))]

    factor_blocks = {}

    for fac in factors:

        dummies = pd.get_dummies(meta_df[fac].astype(str), drop_first=True)

        factor_blocks[fac] = dummies

        if dummies.shape[1] > 0:

            cols_full.append(dummies.to_numpy())

    X_full = np.concatenate(cols_full, axis=1)

    ranks = {"full": int(np.linalg.matrix_rank(X_full))}

    X_minus = {}

    for fac in factors:

        cols = [np.ones((n_samples, 1))]

        for other in factors:

            if other == fac:

                continue

            d = factor_blocks[other]

            if d.shape[1] > 0:

                cols.append(d.to_numpy())

        X_m = np.concatenate(cols, axis=1)

        X_minus[fac] = X_m

        ranks[f"minus_{fac}"] = int(np.linalg.matrix_rank(X_m))

    return X_full, X_minus, ranks

def model_ss_from_G(G_mat, X):

    XtX = X.T @ X

    XtX_inv = np.linalg.pinv(XtX)

    GX = G_mat @ X

    B_loc = X.T @ GX

    SS = float(np.trace(XtX_inv @ B_loc))

    rank = int(np.linalg.matrix_rank(X))

    return SS, rank

def permanova_multi_marginal(G_mat, meta_df, factors,

                             n_perm=999, random_state=42):

    n_samples = G_mat.shape[0]

    assert G_mat.shape[0] == G_mat.shape[1] == meta_df.shape[0]

    SSt = float(np.trace(G_mat))

    X_full, X_minus, ranks = build_design(meta_df, factors)

    SS_full, rank_full = model_ss_from_G(G_mat, X_full)

    df_resid = n_samples - rank_full

    SS_resid = SSt - SS_full

    MS_res = SS_resid / df_resid

    results = {}

    for fac in factors:

        SS_minus, rank_minus = model_ss_from_G(G_mat, X_minus[fac])

        df_fac = rank_full - rank_minus

        SS_fac = SS_full - SS_minus

        R2_fac = SS_fac / SSt if SSt > 0 else np.nan

        MS_fac = SS_fac / df_fac

        F_fac = MS_fac / MS_res if MS_res > 0 else np.nan

        results[fac] = {

            "Df": df_fac,

            "SumOfSqs": SS_fac,

            "R2": R2_fac,

            "F": F_fac,

            "Pr(>F)": np.nan,

        }

    rng = np.random.default_rng(random_state)

    perm_F = {fac: np.zeros(n_perm, dtype=float) for fac in factors}

    print(f"  置换次数: {n_perm} ...  (此步可能略慢)")

    for i in range(n_perm):

        if (i + 1) % 100 == 0:

            print(f"    permutation {i + 1}/{n_perm} ...")

        perm_idx = rng.permutation(n_samples)

        meta_perm = meta_df.iloc[perm_idx].reset_index(drop=True)

        X_full_p, X_minus_p, _ = build_design(meta_perm, factors)

        SS_full_p, rank_full_p = model_ss_from_G(G_mat, X_full_p)

        df_resid_p = n_samples - rank_full_p

        SS_resid_p = SSt - SS_full_p

        MS_res_p = SS_resid_p / df_resid_p

        for fac in factors:

            SS_minus_p, rank_minus_p = model_ss_from_G(G_mat, X_minus_p[fac])

            df_fac_p = rank_full_p - rank_minus_p

            SS_fac_p = SS_full_p - SS_minus_p

            MS_fac_p = SS_fac_p / df_fac_p

            F_p = MS_fac_p / MS_res_p if MS_res_p > 0 else np.nan

            perm_F[fac][i] = F_p

    rows = []

    for fac in factors:

        F_obs = results[fac]["F"]

        perm_vals = perm_F[fac]

        p_val = (1.0 + np.sum(perm_vals >= F_obs)) / (1.0 + n_perm)

        results[fac]["Pr(>F)"] = float(p_val)

        rows.append({

            "Factor": fac,

            "Df": results[fac]["Df"],

            "SumOfSqs": results[fac]["SumOfSqs"],

            "R2": results[fac]["R2"],

            "F": results[fac]["F"],

            "Pr(>F)": results[fac]["Pr(>F)"],

        })

    res_df = pd.DataFrame(rows)

    return res_df

meta_for_perm = df_all[["Organ", "Stage", "Cultivar"]].reset_index(drop=True)

factors = ["Organ", "Stage", "Cultivar"]

permanova_multi_df = permanova_multi_marginal(

    G_mat=G,

    meta_df=meta_for_perm,

    factors=factors,

    n_perm=999,

    random_state=2025,

)

print("\n=== Fig1A 多因子 PERMANOVA 结果 (Python) ===")

print(permanova_multi_df)

multi_out = META_DIR / "Fig1A_PERMANOVA_multifactor_Organ_Stage_Cultivar_Python.csv"

permanova_multi_df.to_csv(multi_out, index=False)

print(f"\n  多因子 PERMANOVA 结果已保存至: {multi_out}")
