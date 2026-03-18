import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

from matplotlib.patches import Patch

from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from matplotlib.lines import Line2D

mpl.rcParams["font.family"] = "Arial"

mpl.rcParams["font.sans-serif"] = ["Arial"]

mpl.rcParams["mathtext.fontset"] = "dejavusans"

mpl.rcParams["pdf.fonttype"] = 42

mpl.rcParams["ps.fonttype"] = 42

mpl.rcParams["svg.fonttype"] = "none"

data_path = "data/GCMS_peel_TIC.csv"

USE_FIXED_RT_RANGE = True

RT_MIN, RT_MAX = 5, 70

stage_order = ["S1", "S2", "S3", "S4"]

cultivar_order = [

    "HMR", "HY", "JG", "JW", "MJY", "MTH", "PG",

    "QNM", "TJY", "XC", "YL", "BTC", "CX", "XY"

]

TIC_SCALE = 1e7

stage_facecolors = {

    "S1": "#9BD7F3",

    "S2": "#D8EEFB",

    "S3": "#FBDDDD",

    "S4": "#F2A1A7",

}

stage_edgecolors = {

    "S1": "#5FAAD8",

    "S2": "#A9C7E8",

    "S3": "#E4B9B9",

    "S4": "#D97C86",

}

stage_linewidth = {s: 0.9 for s in stage_facecolors.keys()}

stage_alpha_fill = {

    "S1": 0.20,

    "S2": 0.26,

    "S3": 0.32,

    "S4": 0.40,

}

print("🔹 正在读取数据:", data_path)

df = pd.read_csv(data_path)

print(f"   总行数: {len(df)}")

print("   列名:", list(df.columns))

df["RT"] = pd.to_numeric(df["RT"], errors="coerce")

df["TIC"] = pd.to_numeric(df["TIC"], errors="coerce")

before = len(df)

df = df.dropna(subset=["RT", "TIC"])

if len(df) < before:

    print(f"   丢弃了 {before - len(df)} 行非数值 RT/TIC")

df_peel = df[df["Organ"].str.lower() == "peel"].copy()

print(f"   果皮数据行数: {len(df_peel)}")

df_peel["Cultivar"] = pd.Categorical(

    df_peel["Cultivar"],

    categories=cultivar_order,

    ordered=True

)

df_peel = df_peel[df_peel["Cultivar"].notna()].copy()

global_rt_min = df_peel["RT"].min()

global_rt_max = df_peel["RT"].max()

print(f"   全部样品 RT 范围: {global_rt_min:.2f} – {global_rt_max:.2f} min")

if USE_FIXED_RT_RANGE:

    print(f"   使用固定 RT 范围: {RT_MIN} – {RT_MAX} min")

else:

    RT_MIN, RT_MAX = global_rt_min, global_rt_max

    print("   使用数据实际 RT 范围")

df_peel["TIC_scaled"] = df_peel["TIC"] / TIC_SCALE

print("   TIC 统一按 1e7 缩放，y 轴将标为 TIC (×10⁷)")

df_peel = df_peel.sort_values(["Cultivar", "Stage", "RT"])

n_cultivars = len(cultivar_order)

print(f"\n🔹 开始绘图，共 {n_cultivars} 个品种")

n_rows, n_cols = 7, 2

fig, axes = plt.subplots(

    nrows=n_rows,

    ncols=n_cols,

    figsize=(10, 8),

    sharex=True,

    sharey=True,

    gridspec_kw={"hspace": 0.0, "wspace": 0.0}

)

axes_flat = axes.ravel()

for i, cultivar in enumerate(cultivar_order):

    ax = axes_flat[i]

    sub_cult = df_peel[df_peel["Cultivar"] == cultivar]

    if sub_cult.empty:

        print(f"   ⚠ 品种 {cultivar} 数据为空，跳过")

        ax.axis("off")

        continue

    print(f"   [{i+1}/{n_cultivars}] 绘制品种 {cultivar} ，行数: {len(sub_cult)}")

    for stage in stage_order:

        sub_stage = sub_cult[sub_cult["Stage"] == stage]

        if sub_stage.empty:

            continue

        sub_stage = sub_stage.sort_values("RT")

        mask = (sub_stage["RT"] >= RT_MIN) & (sub_stage["RT"] <= RT_MAX)

        sub_stage = sub_stage[mask]

        if sub_stage.empty:

            continue

        rt = sub_stage["RT"].values

        tic = sub_stage["TIC_scaled"].values

        ax.plot(

            rt,

            tic,

            color=stage_edgecolors[stage],

            linewidth=stage_linewidth[stage],

            alpha=0.95

        )

    ax.text(

        0.01, 0.92,

        cultivar,

        transform=ax.transAxes,

        ha="left",

        va="top",

        fontsize=14

    )

for idx, ax in enumerate(axes_flat):

    row = idx // n_cols

    col = idx % n_cols

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    if col == 0:

        ax.tick_params(axis="y", labelsize=10, pad=2, labelleft=True, left=True)

    else:

        ax.tick_params(axis="y", labelleft=False, left=False)

    ax.set_xlim(RT_MIN, RT_MAX)

for idx, ax in enumerate(axes_flat):

    row = idx // n_cols

    if row < n_rows - 1:

        ax.tick_params(axis="x", labelbottom=False)

    else:

        ax.tick_params(axis="x", labelsize=10)

fig.text(

    0.04, 0.5,

    r"TIC ($\times 10^7$)",

    ha="center",

    va="center",

    rotation="vertical",

    fontsize=12

)

fig.text(

    0.5, 0.03,

    "Retention time (min)",

    ha="center",

    va="center",

    fontsize=12

)

plt.subplots_adjust(

    left=0.10,

    right=0.97,

    top=0.96,

    bottom=0.08,

    hspace=0.0,

    wspace=0.0

)

legend_handles = [

    Line2D(

        [0], [0],

        color=stage_edgecolors[s],

        linewidth=stage_linewidth[s],

        label=s

    )

    for s in stage_order

]

axes[0, 1].legend(

    handles=legend_handles,

    loc="upper right",

    fontsize=9,

    frameon=True,

    framealpha=1.0,

    borderpad=0.3,

    labelspacing=0.3,

    handlelength=1.0,

    handleheight=0.8,

    borderaxespad=0.35,

    title="Stage",

    title_fontsize=10

)

base = "output/Fig1A_peel_TIC_7x2"

print("\n🔹 保存多种格式的高分辨率图片：")

fig.savefig(base + ".png", dpi=800, bbox_inches="tight", pad_inches=0.02)

print(f"   ✔ {base}.png (dpi=800)")

fig.savefig(base + ".tif", dpi=800, bbox_inches="tight", pad_inches=0.02)

print(f"   ✔ {base}.tif (dpi=800)")

fig.savefig(base + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.02)

print(f"   ✔ {base}.pdf")

fig.savefig(base + ".svg", dpi=600, bbox_inches="tight", pad_inches=0.02)

print(f"   ✔ {base}.svg")

print("\n🔹 绘图完成，在下方显示预览：")

plt.show()

plt.close(fig)
