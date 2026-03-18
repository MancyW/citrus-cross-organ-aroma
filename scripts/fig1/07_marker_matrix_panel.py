import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams["font.family"] = "Arial"

mpl.rcParams["svg.fonttype"] = "none"

mpl.rcParams["pdf.fonttype"] = 42

BASE = Path(".").resolve()

DATA = BASE / "data"

META = DATA / "meta"

OUT  = BASE / "output"

META.mkdir(exist_ok=True, parents=True)

OUT.mkdir(exist_ok=True, parents=True)

PEEL_FILE = DATA / "GCMS_peel.csv"

df = pd.read_csv(PEEL_FILE)

META_COLS = ["SampleID", "Cultivar", "Organ", "Stage", "Batch"]

voc_cols = [c for c in df.columns if c not in META_COLS]

df[voc_cols] = df[voc_cols].fillna(0)

key_vocs = ["Undecanal", "Hexanal", "(+)-(E)-Limonene oxide", "Linalool"]

stage_palette = {

    "S1": "#9BD7F3",

    "S2": "#D8EEFB",

    "S3": "#FBDDDD",

    "S4": "#F2A1A7"

}

for voc in key_vocs:

    df_plot = df[META_COLS + [voc]].copy()

    df_plot["Z"] = (df_plot[voc] - df_plot[voc].mean()) / df_plot[voc].std()

    meta_file = META / f"Fig1G_{voc.replace(' ', '_')}_matrix.csv"

    df_plot.to_csv(meta_file, index=False)

    df_mean = (

        df_plot.groupby(["Cultivar", "Stage"])["Z"]

        .mean()

        .reset_index()

    )

    df_mean["size"] = (df_mean["Z"] - df_mean["Z"].min()) + 0.2

    df_mean["size"] = df_mean["size"] * 260

    fig, ax = plt.subplots(figsize=(14, 3.2))

    for stage in ["S1", "S2", "S3", "S4"]:

        sub = df_mean[df_mean["Stage"] == stage]

        if sub.empty:

            continue

        ax.scatter(

            sub["Cultivar"],

            sub["Stage"],

            s=sub["size"],

            color=stage_palette[stage],

            alpha=0.85,

            edgecolors="black",

            linewidth=0.35,

            rasterized=False

        )

    z_legend = [-0.8, 0, 0.8, 1.6, 2.4]

    sizes_legend = [(z - df_mean["Z"].min()) + 0.2 for z in z_legend]

    sizes_legend = [s * 260 for s in sizes_legend]

    handles = []

    labels = []

    for s, z in zip(sizes_legend, z_legend):

        h = ax.scatter([], [], s=s, color="white", edgecolors="black", rasterized=False)

        handles.append(h)

        labels.append(f"Z = {z}")

    size_leg = ax.legend(

        handles,

        labels,

        title="Z-score (bubble size)",

        frameon=True,

        fontsize=11,

        title_fontsize=12,

        bbox_to_anchor=(1.03, 0.65),

        loc="center left",

        borderpad=1.2,

        labelspacing=2.0,

        handletextpad=1.5

    )

    ax.add_artist(size_leg)

    ax.set_title(voc, fontsize=15, pad=8)

    ax.set_xlabel("Cultivar", fontsize=14)

    ax.set_ylabel("Stage", fontsize=14)

    ax.tick_params(axis="x", rotation=0, labelsize=13)

    ax.tick_params(axis="y", labelsize=13)

    ax.margins(x=0.05, y=0.20)

    ax.set_yticks(["S1", "S2", "S3", "S4"])

    for spine in ["top", "bottom", "left", "right"]:

        ax.spines[spine].set_visible(True)

        ax.spines[spine].set_color("black")

    plt.tight_layout()

    fig.savefig(

        OUT / f"Fig1G_{voc.replace(' ', '_')}_bubble.svg",

        format="svg",

        bbox_inches="tight",

        bbox_extra_artists=[size_leg]

    )

    fig.savefig(

        OUT / f"Fig1G_{voc.replace(' ', '_')}_bubble.pdf",

        format="pdf",

        dpi=600,

        bbox_inches="tight",

        bbox_extra_artists=[size_leg]

    )

    fig.savefig(

        OUT / f"Fig1G_{voc.replace(' ', '_')}_bubble.png",

        dpi=600

    )

    plt.show()
