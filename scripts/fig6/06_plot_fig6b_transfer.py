import numpy as np

import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt

from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]

MID  = ROOT / "intermediate"

OUT  = ROOT / "results"

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["pdf.fonttype"] = 42

plt.rcParams["ps.fonttype"] = 42

plt.rcParams["svg.fonttype"] = "none"

plt.rcParams["font.family"] = "Arial"

STAGE_ORDER = ["S1","S2","S3","S4"]

STAGE2COLOR = {s: plt.cm.tab10(i) for i, s in enumerate(STAGE_ORDER)}

def main(arrows_per_group=2, alpha_pts=0.55, alpha_arrow=0.28, arrow_lw=0.6, seed=1):

    df = pd.read_csv(MID / "sample_desc_2d_relative_v4.csv")

    for col in ["PairID","Cultivar","Stage","Rep","Organ","x","y"]:

        assert col in df.columns, f"Missing {col} in sample_desc_2d_relative_v4.csv"

    df["Stage"] = pd.Categorical(df["Stage"], categories=STAGE_ORDER, ordered=True)

    leaf = df[df["Organ"].eq("Leaf")].copy()

    peel = df[df["Organ"].eq("Peel")].copy()

    pair = leaf.merge(

        peel[["PairID","x","y"]],

        on="PairID",

        how="inner",

        suffixes=("_leaf","_peel")

    )

    pair["dx"] = pair["x_peel"] - pair["x_leaf"]

    pair["dy"] = pair["y_peel"] - pair["y_leaf"]

    pair["dist"] = np.sqrt(pair["dx"]**2 + pair["dy"]**2)

    rng = np.random.default_rng(seed)

    keep_rows = []

    for (cv, st), sub in pair.groupby(["Cultivar","Stage"]):

        sub = sub.copy()

        if len(sub) <= arrows_per_group:

            keep_rows.append(sub)

        else:

            idx = rng.choice(sub.index.values, size=arrows_per_group, replace=False)

            keep_rows.append(sub.loc[idx])

    pair_keep = pd.concat(keep_rows, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), gridspec_kw={"width_ratios":[1.25, 1.0]})

    ax, ax2 = axes

    for organ, marker in [("Leaf","o"),("Peel","^")]:

        sub = df[df["Organ"].eq(organ)].copy()

        colors = sub["Stage"].astype(str).map(STAGE2COLOR)

        ax.scatter(sub["x"], sub["y"], s=16, marker=marker, c=list(colors),

                   alpha=alpha_pts, edgecolors="none")

    for _, r in pair_keep.iterrows():

        c = STAGE2COLOR.get(str(r["Stage"]), (0,0,0,alpha_arrow))

        arr = FancyArrowPatch(

            (r["x_leaf"], r["y_leaf"]),

            (r["x_peel"], r["y_peel"]),

            arrowstyle='-|>',

            mutation_scale=8,

            lw=arrow_lw,

            color=c,

            alpha=alpha_arrow

        )

        ax.add_patch(arr)

    ax.set_xlabel("UMAP-1 (relative-weighted odor-descriptor space)")

    ax.set_ylabel("UMAP-2 (relative-weighted odor-descriptor space)")

    ax.set_title("Fig6B(v4) | Cross-organ perceptual transfer after QC filtering")

    organ_handles = [

        plt.Line2D([0],[0], marker='o', linestyle='', label='Leaf', markersize=6),

        plt.Line2D([0],[0], marker='^', linestyle='', label='Peel', markersize=6),

    ]

    stage_handles = [plt.Line2D([0],[0], marker='s', linestyle='', label=s, markersize=6, color=STAGE2COLOR[s]) for s in STAGE_ORDER]

    leg1 = ax.legend(handles=organ_handles, frameon=False, loc="upper right", title="Organ")

    ax.add_artist(leg1)

    ax.legend(handles=stage_handles, frameon=False, loc="lower right", title="Stage")

    data = []

    labels = []

    for st in STAGE_ORDER:

        vals = pair.loc[pair["Stage"].astype(str).eq(st), "dist"].dropna().to_numpy()

        if len(vals) == 0:

            continue

        data.append(vals); labels.append(st)

    ax2.boxplot(data, labels=labels, showfliers=False)

    for i, vals in enumerate(data, start=1):

        xj = i + (rng.random(len(vals)) - 0.5) * 0.18

        ax2.scatter(xj, vals, s=8, alpha=0.25)

    ax2.set_xlabel("Stage")

    ax2.set_ylabel("||Δ|| in 2D map (Leaf→Peel)")

    ax2.set_title("Displacement magnitude by stage")

    fig.tight_layout()

    fig.savefig(OUT / "Fig6B_transfer_arrows_and_displacement_v4.pdf")

    fig.savefig(OUT / "Fig6B_transfer_arrows_and_displacement_v4.svg")

    pair.to_csv(OUT / "Fig6B_pair_displacement_v4.csv", index=False)

    print("[OK]", OUT / "Fig6B_transfer_arrows_and_displacement_v4.pdf")

if __name__ == "__main__":

    main()
