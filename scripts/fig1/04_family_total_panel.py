import pandas as pd

import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42

mpl.rcParams['ps.fonttype']  = 42

mpl.rcParams['svg.fonttype'] = 'none'

mpl.rcParams['font.family']  = 'Arial'

mpl.rcParams['figure.dpi']   = 300

sns.set_theme(style="white")

BASE_DIR = Path(".")

DATA_DIR = BASE_DIR / "data"

META_DIR = DATA_DIR / "meta"

OUT_DIR  = BASE_DIR / "output"

META_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("▶ Directories ready.")

print("DATA_DIR =", DATA_DIR)

print("META_DIR =", META_DIR)

print("OUT_DIR  =", OUT_DIR)

def save_multiformat(fig, out_path_base):

    fig.savefig(f"{out_path_base}.png", dpi=600)

    fig.savefig(f"{out_path_base}.tif", dpi=600,

                pil_kwargs={"compression": "tiff_lzw"})

    fig.savefig(f"{out_path_base}.pdf")

    fig.savefig(f"{out_path_base}.svg")

    print(f"✔ Saved: {out_path_base}.[png/tif/pdf/svg]")

fixed_palette = {

    "Monoterpenes": "#D8EEFB",

    "Sesquiterpenes": "#FCE6CF",

    "Fatty acid-derived volatiles": "#FBDDDD",

    "Alkanes/Alkenes": "#9BD7F3",

    "Benzenoid/Phenylpropanoid-derived volatiles": "#DCD7EB",

    "Apocarotenoids": "#D5EAD9",

    "N- & heterocycle-containing volatiles": "#7DC69B",

    "Others": "#E6E6E6"

}

print("▶ Loaded updated high-contrast palette.")

print("\n=== Step 4: Load Data ===")

peel = pd.read_csv(DATA_DIR / "GCMS_peel.csv")

cls  = pd.read_csv(DATA_DIR / "compound_classification.csv")

print(f"Peel shape: {peel.shape}")

print(f"Classification shape: {cls.shape}")

print("\nPeel head:")

display(peel.head())

meta_cols = ["SampleID","Cultivar","Organ","Stage","Batch","Rep"]

voc_cols = [c for c in peel.columns if c not in meta_cols]

print(f"\nDetected VOC columns: {len(voc_cols)}")

print("\n=== Step 5: Melt into long format ===")

peel_long = peel.melt(

    id_vars=meta_cols,

    value_vars=voc_cols,

    var_name="Compound",

    value_name="Abundance"

)

print("Melt shape:", peel_long.shape)

display(peel_long.head())

peel_long = peel_long.dropna(subset=["Abundance"])

peel_long = peel_long[peel_long["Abundance"] > 0]

print("After removing zero entries:", peel_long.shape)

print("\n=== Step 6: Merge Classification ===")

family_map = {

    "Monoterpenes": "Monoterpenes",

    "Sesquiterpenes": "Sesquiterpenes",

    "Fatty acid-derived": "Fatty acid-derived volatiles",

    "Fatty-acid-derived": "Fatty acid-derived volatiles",

    "Fatty acid-derived volatiles": "Fatty acid-derived volatiles",

    "Benzenoids": "Benzenoid/Phenylpropanoid-derived volatiles",

    "Benzenoids & phenylpropanoids": "Benzenoid/Phenylpropanoid-derived volatiles",

    "Benzenoid/Phenylpropanoid-derived volatiles":

        "Benzenoid/Phenylpropanoid-derived volatiles",

    "Apocarotenoids": "Apocarotenoids",

    "Alkanes/Alkenes": "Alkanes/Alkenes",

    "N-containing": "N- & heterocycle-containing volatiles",

    "N-containing compounds": "N- & heterocycle-containing volatiles",

    "Miscellaneous heterocycles": "N- & heterocycle-containing volatiles",

    "Others": "Others"

}

df = peel_long.merge(cls, on="Compound", how="left")

missing = df[df["Family"].isna()]["Compound"].unique()

print(f"Missing Family annotations: {len(missing)}")

if len(missing) > 0:

    print(missing)

df["Family"] = df["Family"].map(family_map).fillna("Others")

print("\nFamily classes after normalization:")

print(df["Family"].unique())

display(df.head())

print("\n=== Step 7: Composition Summary ===")

family_sum = (

    df.groupby(["Stage","Family"])["Abundance"]

      .sum()

      .reset_index()

      .pivot(index="Stage", columns="Family", values="Abundance")

      .fillna(0)

)

print("Family_sum:")

display(family_sum)

family_rel = family_sum.div(family_sum.sum(axis=1), axis=0)

print("Family_rel (%):")

display(family_rel)

print("\n=== Step 8: Compute Total Concentration ===")

peel["TotalConcentration"] = peel[voc_cols].sum(axis=1)

print("Row-wise total concentration preview:")

display(peel[["SampleID","Stage","TotalConcentration"]].head())

total_mean = peel.groupby("Stage")["TotalConcentration"].mean()

print("\nStage-mean total VOC (μg/g FW):")

print(total_mean)

scale_factor = 1e4

total_scaled = total_mean / scale_factor

family_sum.to_csv(META_DIR / "Fig1D_peel_family_sum.csv")

family_rel.to_csv(META_DIR / "Fig1D_peel_family_relative.csv")

total_mean.to_csv(META_DIR / "Fig1D_peel_total_concentration.csv")

print("\n✔ Meta files saved.")

print("\n=== Step 10: Plot Fig1D ===")

fig, ax1 = plt.subplots(figsize=(8.5, 7.2))

family_rel.plot(

    kind="bar",

    stacked=True,

    ax=ax1,

    color=[fixed_palette[f] for f in family_rel.columns],

    width=0.85

)

ax1.set_ylabel("Composition (%)", fontsize=15)

ax1.set_xlabel("Developmental stage", fontsize=15)

ax1.tick_params(axis="x", labelsize=14, rotation=0)

ax1.tick_params(axis="y", labelsize=13)

ax1.legend(

    bbox_to_anchor=(0.5, -0.12),

    loc="upper center",

    ncol=3,

    frameon=False,

    fontsize=12

)

ax2 = ax1.twinx()

ax2.plot(

    total_scaled.index,

    total_scaled.values,

    "-o",

    color="#F2A1A7",

    markersize=7,

    linewidth=2.4

)

ax2.set_ylabel(r"Concentration (×10$^4$ μg/g FW)", fontsize=15)

ax2.tick_params(axis="y", labelsize=13)

plt.tight_layout()

out_base = OUT_DIR / "Fig1D_peel_family_stack_with_total_final"

save_multiformat(fig, out_base)

plt.show()

print("\n🎉 Fig1D Completed — vector editable version generated!")
