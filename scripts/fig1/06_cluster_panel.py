import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

mpl.rcParams["font.family"] = "Arial"

mpl.rcParams["pdf.fonttype"] = 42

mpl.rcParams["ps.fonttype"]  = 42

mpl.rcParams["svg.fonttype"] = "none"

BASE_DIR = Path(".").resolve()

DATA_DIR = BASE_DIR / "data"

META_DIR = DATA_DIR / "meta"

OUT_DIR  = BASE_DIR / "output"

META_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "GCMS_peel.csv"

df = pd.read_csv(INPUT_FILE)

meta_cols = ["SampleID", "Cultivar", "Organ", "Stage", "Batch", "Rep"]

voc_cols = [c for c in df.columns if c not in meta_cols]

stage_order = ["S1", "S2", "S3", "S4"]

df["Stage"] = pd.Categorical(df["Stage"], categories=stage_order, ordered=True)

norm_df = df.copy()

eps = 1e-9

for voc in voc_cols:

    norm_df[voc] = norm_df.groupby("Cultivar")[voc].transform(

        lambda x: x / (x.max() + eps)

    )

per_cultivar = (

    norm_df.groupby(["Cultivar", "Stage"])[voc_cols]

           .median()

           .reset_index()

)

dev_mat = (

    per_cultivar.groupby("Stage")[voc_cols]

                .median()

                .loc[stage_order]

)

keep_voc = []

for voc in voc_cols:

    vals = dev_mat[voc].values

    if np.all(vals == 0): continue

    if (vals == 0).sum() >= 3: continue

    if vals.max() < 0.1: continue

    keep_voc.append(voc)

dev_mat_filtered = dev_mat[keep_voc]

scaler = StandardScaler()

z_mat = pd.DataFrame(

    scaler.fit_transform(dev_mat_filtered.T),

    index=dev_mat_filtered.columns,

    columns=stage_order

)

best_k = None

best_score = -999

for k in range(2, 9):

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")

    labels = km.fit_predict(z_mat.values)

    score = silhouette_score(z_mat.values, labels)

    if score > best_score:

        best_k = k

        best_score = score

with open(META_DIR / "Fig1F_peel_bestK.txt", "w") as f:

    f.write(f"Best K = {best_k}, silhouette={best_score:.4f}\n")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")

z_mat["Cluster"] = kmeans.fit_predict(z_mat.values)

z_mat[["Cluster"]].to_csv(META_DIR / "Fig1F_peel_cluster_assignments.csv")

peel_colors = {

    "S1": "#9BD7F3",

    "S2": "#D8EEFB",

    "S3": "#FBDDDD",

    "S4": "#F2A1A7"

}

fig, axes = plt.subplots(best_k, 1, figsize=(6, 2.8*best_k), sharey=False)

x = np.arange(len(stage_order))

x_dense = np.linspace(x.min(), x.max(), 200)

def smooth_y(y):

    spline = make_interp_spline(x, y, k=3)

    return spline(x_dense)

for k in range(best_k):

    ax = axes[k]

    sub = z_mat[z_mat["Cluster"] == k].drop(columns=["Cluster"])

    mean_curve = sub.mean(axis=0).values

    ymin = sub.min(axis=0).values

    ymax = sub.max(axis=0).values

    mean_s = smooth_y(mean_curve)

    ymin_s = smooth_y(ymin)

    ymax_s = smooth_y(ymax)

    ax.fill_between(x_dense, ymin_s, ymax_s,

                    color="#D8EEFB", alpha=0.45)

    for _, row in sub.iterrows():

        ax.plot(x, row.values, color="#FBDDDD", alpha=0.7, linewidth=0.8)

    ax.plot(x_dense, mean_s, color="#F2A1A7", linewidth=2.2)

    for i, stg in enumerate(stage_order):

        ax.scatter(

            i, mean_curve[i],

            s=45, color=peel_colors[stg],

            edgecolor="black", linewidth=0.6, zorder=10

        )

    ax.set_xticks(x)

    ax.set_xticklabels(stage_order, fontsize=11)

    ax.set_xlim(-0.2, 3.2)

    ax.grid(False)

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.set_title(f"Cluster {k+1} (n={sub.shape[0]})", fontsize=12)

fig.text(0.04, 0.5, "Z-score", rotation="vertical", fontsize=12)

axes[-1].set_xlabel("Stage (S1–S4)", fontsize=11)

plt.tight_layout(rect=[0.06, 0.03, 1, 1])

for ext in ["png", "svg", "pdf", "tif"]:

    fig.savefig(

        OUT_DIR / f"Fig1F_peel_kmeans.{ext}",

        dpi=600,

        bbox_inches="tight"

    )

plt.show()
