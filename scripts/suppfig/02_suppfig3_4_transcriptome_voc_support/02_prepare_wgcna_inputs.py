from pathlib import Path
import os
import pandas as pd
import numpy as np
import re

project = Path(os.environ.get("PROJECT_DIR", ".")).resolve()

vst_file = project / "analysis/rnaseq_qc/RNAseq_416samples.vst_matrix.tsv"
meta_file = project / "counts/matrix/RNAseq_416samples.sample_metadata.tsv"

leaf_voc_file = project / "analysis/voc_traits/leaf_voc_clip0_log1p_mean_52profiles.tsv"
peel_voc_file = project / "analysis/voc_traits/peel_voc_clip0_log1p_mean_52profiles.tsv"
leaf_key_file = project / "analysis/voc_traits/leaf_key_voc_clip0_log1p_mean_52profiles.tsv"
peel_key_file = project / "analysis/voc_traits/peel_key_voc_clip0_log1p_mean_52profiles.tsv"

out_dir = project / "analysis/wgcna_inputs"
out_dir.mkdir(parents=True, exist_ok=True)

top_n_genes = 12000

def make_safe_name(x):
    x = str(x)
    x = x.replace("β", "beta")
    x = x.replace("α", "alpha")
    x = x.replace("γ", "gamma")
    x = re.sub(r"[^A-Za-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    if x == "":
        x = "trait"
    if re.match(r"^[0-9]", x):
        x = "X_" + x
    return x

def make_unique(names):
    seen = {}
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 1
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
    return out

def find_sample_col(meta, vst_cols):
    candidates = ["sample_id", "sample", "SampleID", "Sample", "sample_name", "SampleName"]
    for c in candidates:
        if c in meta.columns and set(meta[c].astype(str)).issubset(set(vst_cols)):
            return c

    # fallback: choose the column with the largest overlap with VST columns
    best_col = None
    best_overlap = -1
    for c in meta.columns:
        vals = set(meta[c].astype(str))
        overlap = len(vals & set(vst_cols))
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = c

    if best_overlap <= 0:
        raise ValueError("Could not identify sample column in metadata.")
    return best_col

print("Reading VST matrix:", vst_file)
vst = pd.read_csv(vst_file, sep="\t")

gene_col = vst.columns[0]
vst = vst.rename(columns={gene_col: "Geneid"})
vst["Geneid"] = vst["Geneid"].astype(str)
vst = vst.set_index("Geneid")

print("VST shape genes x samples:", vst.shape)

print("Reading sample metadata:", meta_file)
meta = pd.read_csv(meta_file, sep="\t", dtype=str)
sample_col = find_sample_col(meta, vst.columns)
print("Detected sample column:", sample_col)

# Normalize metadata column names needed downstream
col_map = {}
for c in meta.columns:
    lc = c.lower()
    if lc == "organ":
        col_map[c] = "organ"
    elif lc == "cultivar":
        col_map[c] = "cultivar"
    elif lc == "stage":
        col_map[c] = "stage"
    elif lc in ["replicate", "rep", "bio_rep"]:
        col_map[c] = "replicate"

meta = meta.rename(columns=col_map)
required = ["organ", "cultivar", "stage"]
missing_required = [c for c in required if c not in meta.columns]
if missing_required:
    raise ValueError(f"Metadata missing required columns after normalization: {missing_required}")

if "replicate" not in meta.columns:
    meta["replicate"] = "NA"

meta["sample_id"] = meta[sample_col].astype(str)
meta["organ"] = meta["organ"].astype(str)
meta["cultivar"] = meta["cultivar"].astype(str)
meta["stage"] = meta["stage"].astype(str)
meta["replicate"] = meta["replicate"].astype(str)
meta["group_id"] = meta["cultivar"] + "_" + meta["stage"]

# Keep only samples present in VST
meta = meta[meta["sample_id"].isin(vst.columns)].copy()

valid_cultivars = sorted(meta["cultivar"].unique())
valid_stages = ["S1", "S2", "S3", "S4"]
rna_profiles = sorted((meta["cultivar"] + "_" + meta["stage"]).unique())

print("Metadata rows after VST matching:", meta.shape[0])
print("RNA profiles:", len(rna_profiles))
print("Cultivars:", valid_cultivars)

# Basic finite check
vst = vst.apply(pd.to_numeric, errors="coerce")
bad_genes = vst.index[vst.isna().any(axis=1)].tolist()
if bad_genes:
    print("Genes with NA in VST, removing:", len(bad_genes))
    vst = vst.drop(index=bad_genes)

if np.isinf(vst.values).sum() > 0:
    raise ValueError("VST matrix contains Inf values.")

def prepare_organ_expression(organ):
    organ_meta = meta[meta["organ"].str.lower() == organ.lower()].copy()
    organ_meta = organ_meta.sort_values(["cultivar", "stage", "replicate", "sample_id"])

    sample_ids = organ_meta["sample_id"].tolist()
    expr = vst[sample_ids].T.copy()
    expr.index.name = "sample_id"

    # Top variable genes within this organ
    variances = expr.var(axis=0, ddof=1).sort_values(ascending=False)
    selected_genes = variances.head(min(top_n_genes, len(variances))).index.tolist()

    expr_top = expr[selected_genes].copy()

    # Sample-level matrix for WGCNA
    sample_meta_out = organ_meta[["sample_id", "group_id", "cultivar", "stage", "replicate", "organ"]].copy()
    sample_expr_out = pd.concat(
        [sample_meta_out.set_index("sample_id"), expr_top],
        axis=1
    ).reset_index()

    sample_expr_file = out_dir / f"{organ.lower()}_vst_top{len(selected_genes)}_sample_expression.tsv"
    sample_expr_out.to_csv(sample_expr_file, sep="\t", index=False)

    # Pure matrix for WGCNA: rows = samples, columns = genes
    pure_sample_file = out_dir / f"{organ.lower()}_vst_top{len(selected_genes)}_sample_expression_matrix_only.tsv"
    expr_top.insert(0, "sample_id", expr_top.index)
    expr_top.to_csv(pure_sample_file, sep="\t", index=False)
    expr_top = expr_top.drop(columns=["sample_id"])

    # Profile mean expression: 52 cultivar-stage profiles
    expr_with_group = expr_top.copy()
    expr_with_group["group_id"] = organ_meta.set_index("sample_id").loc[expr_top.index, "group_id"]

    profile_expr = expr_with_group.groupby("group_id")[selected_genes].mean()
    missing_profiles = sorted(set(rna_profiles) - set(profile_expr.index))
    if missing_profiles:
        raise ValueError(f"{organ}: missing profiles in profile expression: {missing_profiles}")

    profile_expr = profile_expr.loc[rna_profiles]
    profile_expr_out = profile_expr.copy()
    profile_expr_out.insert(0, "group_id", profile_expr_out.index)

    profile_expr_file = out_dir / f"{organ.lower()}_vst_top{len(selected_genes)}_profile_mean_expression_52profiles.tsv"
    profile_expr_out.to_csv(profile_expr_file, sep="\t", index=False)

    # Selected gene variance table
    var_file = out_dir / f"{organ.lower()}_selected_top_variable_genes.tsv"
    pd.DataFrame({
        "Geneid": variances.index,
        "variance": variances.values,
        "selected_top_variable": [g in selected_genes for g in variances.index]
    }).to_csv(var_file, sep="\t", index=False)

    print(f"\n{organ}:")
    print("  RNA-seq samples:", len(sample_ids))
    print("  selected genes:", len(selected_genes))
    print("  sample matrix:", sample_expr_file, sample_expr_out.shape)
    print("  pure WGCNA matrix:", pure_sample_file, expr_top.shape)
    print("  profile mean matrix:", profile_expr_file, profile_expr_out.shape)

    return {
        "organ": organ,
        "n_samples": len(sample_ids),
        "n_selected_genes": len(selected_genes),
        "sample_expr_file": str(sample_expr_file),
        "pure_sample_file": str(pure_sample_file),
        "profile_expr_file": str(profile_expr_file),
        "var_file": str(var_file),
    }

leaf_summary = prepare_organ_expression("Leaf")
peel_summary = prepare_organ_expression("Peel")

# Prepare VOC trait matrices with safe names
def read_voc_matrix(path, prefix):
    df = pd.read_csv(path, sep="\t")
    if "group_id" not in df.columns:
        raise ValueError(f"{path} missing group_id column.")
    df = df.set_index("group_id").loc[rna_profiles]
    value_cols = list(df.columns)

    safe_cols_raw = [prefix + "__" + make_safe_name(c) for c in value_cols]
    safe_cols = make_unique(safe_cols_raw)

    mapping = pd.DataFrame({
        "prefix": prefix,
        "original_trait": value_cols,
        "safe_trait": safe_cols,
    })

    df.columns = safe_cols
    df.insert(0, "group_id", df.index)
    return df.reset_index(drop=True), mapping

leaf_voc_all, map_leaf_all = read_voc_matrix(leaf_voc_file, "LeafVOC")
peel_voc_all, map_peel_all = read_voc_matrix(peel_voc_file, "PeelVOC")
leaf_voc_key, map_leaf_key = read_voc_matrix(leaf_key_file, "LeafVOC")
peel_voc_key, map_peel_key = read_voc_matrix(peel_key_file, "PeelVOC")

profile_meta = pd.DataFrame({"group_id": rna_profiles})
profile_meta["cultivar"] = profile_meta["group_id"].str.replace(r"_(S[1-4])$", "", regex=True)
profile_meta["stage"] = profile_meta["group_id"].str.extract(r"_(S[1-4])$")

# all VOC trait matrix
trait_all = profile_meta.copy()
trait_all = trait_all.merge(leaf_voc_all, on="group_id")
trait_all = trait_all.merge(peel_voc_all, on="group_id")

trait_all_file = out_dir / "traits_leaf_peel_all_voc_clip0_log1p_52profiles.safe.tsv"
trait_all.to_csv(trait_all_file, sep="\t", index=False)

# key VOC trait matrix
trait_key = profile_meta.copy()
trait_key = trait_key.merge(leaf_voc_key, on="group_id")
trait_key = trait_key.merge(peel_voc_key, on="group_id")

trait_key_file = out_dir / "traits_leaf_peel_key_voc_clip0_log1p_52profiles.safe.tsv"
trait_key.to_csv(trait_key_file, sep="\t", index=False)

trait_map = pd.concat(
    [map_leaf_all, map_peel_all],
    axis=0,
    ignore_index=True
).drop_duplicates()

trait_map_file = out_dir / "voc_trait_name_map.safe.tsv"
trait_map.to_csv(trait_map_file, sep="\t", index=False)

# stage/cultivar design metadata
profile_meta_file = out_dir / "profile_metadata_52profiles.tsv"
profile_meta.to_csv(profile_meta_file, sep="\t", index=False)

summary = pd.DataFrame([leaf_summary, peel_summary])
summary_file = out_dir / "wgcna_input_preparation_summary.tsv"
summary.to_csv(summary_file, sep="\t", index=False)

print("\nTrait matrices:")
print("  all VOC traits:", trait_all_file, trait_all.shape)
print("  key VOC traits:", trait_key_file, trait_key.shape)
print("  trait name map:", trait_map_file, trait_map.shape)
print("  profile metadata:", profile_meta_file, profile_meta.shape)

print("\nSummary:")
print(summary.to_string(index=False))
print("\nDone.")
