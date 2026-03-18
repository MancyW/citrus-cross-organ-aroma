import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.cross_decomposition import PLSRegression

from sklearn.model_selection import KFold, cross_val_predict

from matplotlib.patches import Ellipse, Patch

plt.rcParams["font.family"] = "Arial"

mpl.rcParams["pdf.fonttype"] = 42

mpl.rcParams["ps.fonttype"] = 42

mpl.rcParams["svg.fonttype"] = "none"

def add_confidence_ellipse(x, y, ax, n_std=2.0, facecolor="none", **kwargs):

    x = np.asarray(x)

    y = np.asarray(y)

    if x.size <= 2:

        return

    cov = np.cov(x, y)

    vals, vecs = np.linalg.eigh(cov)

    order = vals.argsort()[::-1]

    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2 * n_std * np.sqrt(vals)

    ellipse = Ellipse(

        xy=(np.mean(x), np.mean(y)),

        width=width,

        height=height,

        angle=theta,

        facecolor=facecolor,

        **kwargs

    )

    ax.add_patch(ellipse)

def main():

    input_path = "data/GCMS_peel.csv"

    print(f"🔹 正在读取数据: {input_path}")

    df = pd.read_csv(input_path)

    print(f"   总样本数: {df.shape[0]}")

    print(f"   总列数: {df.shape[1]}")

    print(f"   列名预览: {list(df.columns[:6])} ...")

    print("   Organ 分布:")

    print("  ", df["Organ"].value_counts().to_dict())

    print("\n🔹 构建代谢物矩阵并预处理 (StandardScaler)")

    df_peel = df[df["Organ"] == "Peel"].copy()

    print(f"   仅保留果皮样本后的样本数: {df_peel.shape[0]}")

    X = df_peel.iloc[:, 6:]

    n_features_raw = X.shape[1]

    print(f"   原始 VOC 峰数量: {n_features_raw}")

    X = X.loc[:, X.std(axis=0) > 0]

    n_features_used = X.shape[1]

    print(f"   去除方差为 0 的峰后保留: {n_features_used} 个峰")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("   已完成标准化 (mean=0, var=1)")

    stage_order = ["S1", "S2", "S3", "S4"]

    peel_stage_colors = {

        "S1": "#9BD7F3",

        "S2": "#D8EEFB",

        "S3": "#FBDDDD",

        "S4": "#F2A1A7",

    }

    print("\n🔹 运行 PLS-DA (Stage 作为 Y)")

    try:

        enc = OneHotEncoder(sparse_output=False, categories=[stage_order])

    except TypeError:

        enc = OneHotEncoder(sparse=False, categories=[stage_order])

    Y = enc.fit_transform(df_peel[["Stage"]])

    pls = PLSRegression(n_components=2)

    T, U = pls.fit_transform(X_scaled, Y)

    scores_df = pd.DataFrame(T, columns=["LV1", "LV2"])

    scores_df["Stage"] = df_peel["Stage"].values

    print("   各 Stage 样本数量:")

    for st in stage_order:

        n_st = (scores_df["Stage"] == st).sum()

        print(f"   - {st}: {n_st} 个样本")

    centroids = (

        scores_df

        .groupby("Stage")[["LV1", "LV2"]]

        .mean()

        .loc[stage_order]

    )

    print("\n   Stage 在 LV1 上的均值 (可视为发育潜变量):")

    for st, (lv1, lv2) in centroids.iterrows():

        print(f"   - {st}: LV1={lv1:.3f}, LV2={lv2:.3f}")

    print("\n🔹 计算 R²X, R²Y, Q²")

    Y_pred = pls.predict(X_scaled)

    tss_y = ((Y - Y.mean(axis=0)) ** 2).sum()

    sse_y = ((Y - Y_pred) ** 2).sum()

    R2Y = 1 - sse_y / tss_y

    T_scores = pls.x_scores_

    P_loadings = pls.x_loadings_

    X_hat = T_scores @ P_loadings.T

    ssx = (X_scaled ** 2).sum()

    resx = ((X_scaled - X_hat) ** 2).sum()

    R2X = 1 - resx / ssx

    kf = KFold(n_splits=7, shuffle=True, random_state=42)

    pls_cv = PLSRegression(n_components=2)

    Y_cv_pred = cross_val_predict(pls_cv, X_scaled, Y, cv=kf)

    press = ((Y - Y_cv_pred) ** 2).sum()

    Q2 = 1 - press / tss_y

    print(f"   R²X(cum) = {R2X:.3f}")

    print(f"   R²Y(cum) = {R2Y:.3f}")

    print(f"   Q²(cum)  = {Q2:.3f}")

    print("\n🔹 开始绘制 Fig1B (Peel PLS-DA)")

    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=300)

    for st in stage_order:

        sub = scores_df[scores_df["Stage"] == st]

        ax.scatter(

            sub["LV1"],

            sub["LV2"],

            s=15,

            c=peel_stage_colors[st],

            edgecolor="k",

            linewidth=0.3,

            alpha=0.85,

            label=st,

        )

        add_confidence_ellipse(

            sub["LV1"], sub["LV2"], ax,

            n_std=2.0,

            edgecolor=peel_stage_colors[st],

            facecolor=peel_stage_colors[st],

            alpha=0.18,

            linewidth=0.8,

        )

    ax.plot(

        centroids["LV1"],

        centroids["LV2"],

        "-o",

        color="k",

        linewidth=1.0,

        markersize=4,

    )

    ax.axhline(0, color="lightgray", linewidth=0.5, zorder=0)

    ax.axvline(0, color="lightgray", linewidth=0.5, zorder=0)

    ax.set_xlabel("LV1", fontsize=8)

    ax.set_ylabel("LV2", fontsize=8)

    ax.tick_params(axis="both", labelsize=7)

    legend_handles = [

        Patch(

            facecolor=peel_stage_colors[s],

            edgecolor="none",

            alpha=0.85,

            label=s

        ) for s in stage_order

    ]

    leg = ax.legend(

        handles=legend_handles,

        loc="upper right",

        fontsize=7,

        frameon=True,

        framealpha=1.0,

        borderpad=0.4,

        title="Stage",

        title_fontsize=8

    )

    text_str = (

        f"R²X(cum) = {R2X:.2f}\n"

        f"R²Y(cum) = {R2Y:.2f}\n"

        f"Q²(cum)  = {Q2:.2f}"

    )

    ax.text(

        0.03, 0.03, text_str,

        transform=ax.transAxes,

        ha="left", va="bottom",

        fontsize=6,

        linespacing=1.4,

    )

    plt.tight_layout()

    os.makedirs("output", exist_ok=True)

    base = os.path.join("output", "Fig1B_peel_PLSDA")

    print("\n🔹 保存多种格式的高分辨率图片：")

    fig.savefig(base + ".png", dpi=800)

    print(f"   ✔ {base}.png (dpi=800)")

    fig.savefig(base + ".tif", dpi=800)

    print(f"   ✔ {base}.tif (dpi=800)")

    fig.savefig(base + ".pdf")

    print(f"   ✔ {base}.pdf")

    fig.savefig(base + ".svg")

    print(f"   ✔ {base}.svg")

    print("\n🔹 绘图完成，在本地/Notebook 下将显示预览：")

    plt.show()

    plt.close()

    print("\n✅ 全部完成。")

if __name__ == "__main__":

    main()
