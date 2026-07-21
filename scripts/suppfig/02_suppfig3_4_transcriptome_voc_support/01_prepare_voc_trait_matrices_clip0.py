from pathlib import Path
import os
import pandas as pd
import numpy as np

project = Path(os.environ.get("PROJECT_DIR", ".")).resolve()

leaf_file = project / "voc/raw/GCMS_leaf.raw.csv"
peel_file = project / "voc/raw/GCMS_peel.raw.csv"
rna_meta_file = project / "counts/matrix/RNAseq_416samples.sample_metadata.tsv"

out_dir = project / "analysis/voc_traits"
out_dir.mkdir(parents=True, exist_ok=True)

rna_meta = pd.read_csv(rna_meta_file, sep="\t", dtype=str)
valid_cultivars = sorted(rna_meta["cultivar"].unique())
valid_stages = ["S1", "S2", "S3", "S4"]
rna_profiles = sorted((rna_meta["cultivar"] + "_" + rna_meta["stage"]).unique())

meta_cols = ["SampleID", "Cultivar", "Organ", "Stage", "Batch", "Rep"]

key_voc_candidates = [
    "β-Ocimene",
    "(E)-β-Ocimene",
    "trans-β-Ocimene",
    "Linalool",
    "Decanal",
    "Citronellol",
    "β-Myrcene",
    "Perillaldehyde",
    "D-Limonene",
    "Benzeneacetaldehyde",
    "Perilla alcohol",
    "Citronellal",
    "Nonanal",
    "Octanal",
    "Undecanal",
    "Dodecanal",
]

all_negative_cells = []

def read_prepare_clip0(path, expected_organ):
    df = pd.read_csv(path)

    missing_meta = [c for c in meta_cols if c not in df.columns]
    if missing_meta:
        raise ValueError(f"{path} missing metadata columns: {missing_meta}")

    for c in meta_cols:
        df[c] = df[c].astype(str)

    df["group_id"] = df["Cultivar"] + "_" + df["Stage"]

    df = df[
        df["Cultivar"].isin(valid_cultivars) &
        df["Stage"].isin(valid_stages)
    ].copy()

    voc_cols = [c for c in df.columns if c not in meta_cols + ["group_id"]]

    for c in voc_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[voc_cols] = df[voc_cols].fillna(0)

    # Record negative cells before clipping
    neg_mask = df[voc_cols] < 0
    if neg_mask.values.any():
        neg_positions = np.where(neg_mask.values)
        for row_i, col_i in zip(*neg_positions):
            voc = voc_cols[col_i]
            all_negative_cells.append({
                "Organ": expected_organ,
                "SampleID": df.iloc[row_i]["SampleID"],
                "Cultivar": df.iloc[row_i]["Cultivar"],
                "Stage": df.iloc[row_i]["Stage"],
                "Rep": df.iloc[row_i]["Rep"],
                "VOC": voc,
                "original_value": df.iloc[row_i][voc],
            })

    n_negative = int((df[voc_cols] < 0).sum().sum())
    min_value_before = float(np.nanmin(df[voc_cols].values))
    max_value_before = float(np.nanmax(df[voc_cols].values))

    # Formal correction: VOC abundance cannot be negative
    df_clip = df.copy()
    df_clip[voc_cols] = df_clip[voc_cols].clip(lower=0)

    min_value_after = float(np.nanmin(df_clip[voc_cols].values))
    max_value_after = float(np.nanmax(df_clip[voc_cols].values))

    log_df = df_clip.copy()
    log_df[voc_cols] = np.log1p(log_df[voc_cols])

    if log_df[voc_cols].isna().sum().sum() != 0:
        raise ValueError(f"{expected_organ}: NaN generated after clip0 log1p.")
    if np.isinf(log_df[voc_cols].values).sum() != 0:
        raise ValueError(f"{expected_organ}: Inf generated after clip0 log1p.")

    rep_counts = (
        df_clip.groupby(["Cultivar", "Stage", "Organ"])
        .size()
        .reset_index(name="n_voc_reps")
        .sort_values(["Cultivar", "Stage", "Organ"])
    )

    mean_df = (
        log_df.groupby(["Cultivar", "Stage", "Organ", "group_id"], as_index=False)[voc_cols]
        .mean()
        .sort_values(["Cultivar", "Stage"])
        .reset_index(drop=True)
    )

    voc_profiles = sorted(mean_df["group_id"].unique())
    missing_from_voc = sorted(set(rna_profiles) - set(voc_profiles))
    extra_in_voc = sorted(set(voc_profiles) - set(rna_profiles))

    print(f"\n{expected_organ}")
    print("Rows:", df.shape[0])
    print("VOC columns:", len(voc_cols))
    print("Mean profiles:", len(voc_profiles))
    print("Negative values before clip:", n_negative)
    print("Min before clip:", min_value_before)
    print("Max before clip:", max_value_before)
    print("Min after clip:", min_value_after)
    print("Max after clip:", max_value_after)
    print("Missing RNA profiles in VOC:", missing_from_voc)
    print("Extra VOC profiles:", extra_in_voc)

    if missing_from_voc:
        raise ValueError(f"{expected_organ}: missing profiles {missing_from_voc}")

    return df_clip, log_df, mean_df, rep_counts, voc_cols

leaf_clip, leaf_log, leaf_mean, leaf_rep_counts, leaf_voc_cols = read_prepare_clip0(leaf_file, "Leaf")
peel_clip, peel_log, peel_mean, peel_rep_counts, peel_voc_cols = read_prepare_clip0(peel_file, "Peel")

negative_cells_df = pd.DataFrame(all_negative_cells)
negative_cells_file = out_dir / "voc_negative_cells_clipped_to_zero.tsv"
negative_cells_df.to_csv(negative_cells_file, sep="\t", index=False)

shared_voc_cols = sorted(set(leaf_voc_cols) & set(peel_voc_cols))
leaf_only_cols = sorted(set(leaf_voc_cols) - set(peel_voc_cols))
peel_only_cols = sorted(set(peel_voc_cols) - set(leaf_voc_cols))

print("\nShared VOC columns:", len(shared_voc_cols))
print("Leaf-only VOC columns:", len(leaf_only_cols))
print("Peel-only VOC columns:", len(peel_only_cols))
print("Negative cells clipped to zero:", len(negative_cells_df))

def export_matrix(mean_df, voc_cols, organ_name):
    tmp = mean_df.set_index("group_id").loc[rna_profiles]

    meta = tmp[["Cultivar", "Stage", "Organ"]].copy()
    mat = tmp[voc_cols].copy()

    mat.insert(0, "group_id", mat.index)
    meta.insert(0, "group_id", meta.index)

    mat_out = out_dir / f"{organ_name.lower()}_voc_clip0_log1p_mean_52profiles.tsv"
    meta_out = out_dir / f"{organ_name.lower()}_voc_clip0_log1p_mean_52profiles_metadata.tsv"

    mat.to_csv(mat_out, sep="\t", index=False)
    meta.to_csv(meta_out, sep="\t", index=False)

    print(f"Wrote {organ_name} matrix:", mat_out, mat.shape)
    print(f"Wrote {organ_name} metadata:", meta_out, meta.shape)

    return mat, meta

leaf_matrix, leaf_meta = export_matrix(leaf_mean, shared_voc_cols, "Leaf")
peel_matrix, peel_meta = export_matrix(peel_mean, shared_voc_cols, "Peel")

leaf_rep_counts.to_csv(out_dir / "leaf_voc_clip0_replicate_counts_by_group.tsv", sep="\t", index=False)
peel_rep_counts.to_csv(out_dir / "peel_voc_clip0_replicate_counts_by_group.tsv", sep="\t", index=False)

available_key_vocs = [c for c in key_voc_candidates if c in shared_voc_cols]
missing_key_vocs = [c for c in key_voc_candidates if c not in shared_voc_cols]

print("\nAvailable key VOCs:", available_key_vocs)
print("Missing key VOCs:", missing_key_vocs)

leaf_key = leaf_matrix[["group_id"] + available_key_vocs].copy()
peel_key = peel_matrix[["group_id"] + available_key_vocs].copy()

leaf_key.to_csv(out_dir / "leaf_key_voc_clip0_log1p_mean_52profiles.tsv", sep="\t", index=False)
peel_key.to_csv(out_dir / "peel_key_voc_clip0_log1p_mean_52profiles.tsv", sep="\t", index=False)

peel_s4 = peel_matrix[peel_matrix["group_id"].str.endswith("_S4")].copy()
peel_s4.insert(1, "cultivar", peel_s4["group_id"].str.replace("_S4", "", regex=False))
peel_s4.to_csv(out_dir / "peel_mature_S4_voc_clip0_log1p_13cultivars.tsv", sep="\t", index=False)

peel_s4_key = peel_s4[["group_id", "cultivar"] + available_key_vocs].copy()
peel_s4_key.to_csv(out_dir / "peel_mature_S4_key_voc_clip0_log1p_13cultivars.tsv", sep="\t", index=False)

# Long key VOC table
long_rows = []
for organ_name, mat in [("Leaf", leaf_key), ("Peel", peel_key)]:
    tmp = mat.copy()
    tmp["Cultivar"] = tmp["group_id"].str.replace("_S[1-4]$", "", regex=True)
    tmp["Stage"] = tmp["group_id"].str.extract(r"_(S[1-4])$")
    for voc in available_key_vocs:
        for _, r in tmp.iterrows():
            long_rows.append({
                "Organ": organ_name,
                "group_id": r["group_id"],
                "Cultivar": r["Cultivar"],
                "Stage": r["Stage"],
                "VOC": voc,
                "clip0_log1p_mean_abundance": r[voc],
            })

long_df = pd.DataFrame(long_rows)
long_df.to_csv(out_dir / "key_voc_clip0_log1p_mean_long.tsv", sep="\t", index=False)

summary = pd.DataFrame([
    {
        "matrix": "leaf",
        "n_raw_rows_filtered": leaf_clip.shape[0],
        "n_mean_profiles": leaf_matrix.shape[0],
        "n_shared_voc_traits": len(shared_voc_cols),
        "n_available_key_vocs": len(available_key_vocs),
        "n_negative_cells_clipped": int((negative_cells_df["Organ"] == "Leaf").sum()) if not negative_cells_df.empty else 0,
        "n_nan_final": int(leaf_matrix.drop(columns=["group_id"]).isna().sum().sum()),
    },
    {
        "matrix": "peel",
        "n_raw_rows_filtered": peel_clip.shape[0],
        "n_mean_profiles": peel_matrix.shape[0],
        "n_shared_voc_traits": len(shared_voc_cols),
        "n_available_key_vocs": len(available_key_vocs),
        "n_negative_cells_clipped": int((negative_cells_df["Organ"] == "Peel").sum()) if not negative_cells_df.empty else 0,
        "n_nan_final": int(peel_matrix.drop(columns=["group_id"]).isna().sum().sum()),
    },
])

summary_file = out_dir / "voc_clip0_log1p_trait_preparation_summary.tsv"
summary.to_csv(summary_file, sep="\t", index=False)

print("\nWrote negative cell record:", negative_cells_file)
print("Wrote summary:", summary_file)
print(summary.to_string(index=False))
