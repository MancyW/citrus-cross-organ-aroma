from pathlib import Path
import os
import pandas as pd
import numpy as np
import shutil

project = Path(os.environ.get("PROJECT_DIR", ".")).resolve()
out_dir = project / "analysis/final_evidence_package"
out_dir.mkdir(parents=True, exist_ok=True)

def read_tsv(path, required=False):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        print(f"[WARN] Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")

def write_tsv(df, name):
    path = out_dir / name
    df.to_csv(path, sep="\t", index=False)
    print(f"Wrote {path} {df.shape}")
    return path

def safe_num(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def add_metric(rows, category, metric, value, interpretation=""):
    rows.append({
        "category": category,
        "metric": metric,
        "value": value,
        "interpretation": interpretation
    })

def df_to_md(df, max_rows=20):
    if df is None or df.empty:
        return "_No data available._\n"
    d = df.head(max_rows).copy()
    for c in d.columns:
        d[c] = d[c].astype(str).str.replace("\n", " ", regex=False)
        d[c] = d[c].str.slice(0, 120)
    header = "| " + " | ".join(d.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(d.columns)) + " |"
    rows = []
    for _, r in d.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in d.columns) + " |")
    return "\n".join([header, sep] + rows) + "\n"

def add_abs_sort(df, cor_col="cor", fdr_col="padj_BH"):
    if df.empty:
        return df
    out = df.copy()
    if cor_col in out.columns:
        out[cor_col] = pd.to_numeric(out[cor_col], errors="coerce")
        out["abs_cor"] = out[cor_col].abs()
    if fdr_col in out.columns:
        out[fdr_col] = pd.to_numeric(out[fdr_col], errors="coerce")
        out = out.sort_values([fdr_col, "abs_cor"], ascending=[True, False], na_position="last")
    elif "abs_cor" in out.columns:
        out = out.sort_values("abs_cor", ascending=False)
    return out

def load_trait_map():
    trait_map_file = project / "analysis/wgcna_inputs/voc_trait_name_map.safe.tsv"
    m = read_tsv(trait_map_file)
    if m.empty or not {"safe_trait", "original_trait"}.issubset(m.columns):
        return {}
    return dict(zip(m["safe_trait"], m["original_trait"]))

trait_map = load_trait_map()

def annotate_trait(df, col="right"):
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out["trait_original"] = out[col].map(trait_map).fillna(out[col])
    return out

# ------------------------------------------------------------------
# 1. Key metrics
# ------------------------------------------------------------------

metrics = []

voc_summary = read_tsv(project / "analysis/voc_traits/voc_clip0_log1p_trait_preparation_summary.tsv")
if not voc_summary.empty:
    for _, r in voc_summary.iterrows():
        add_metric(
            metrics,
            "VOC trait matrix",
            f"{r['matrix']}: mean profiles",
            r.get("n_mean_profiles", ""),
            "VOC traits were averaged to 52 cultivar-stage profiles."
        )
        add_metric(
            metrics,
            "VOC trait matrix",
            f"{r['matrix']}: shared VOC traits",
            r.get("n_shared_voc_traits", ""),
            "Shared leaf/peel VOC traits used for downstream analyses."
        )
        add_metric(
            metrics,
            "VOC trait matrix",
            f"{r['matrix']}: key VOC traits",
            r.get("n_available_key_vocs", ""),
            "Key VOC traits available for module-VOC analysis."
        )
        add_metric(
            metrics,
            "VOC trait matrix",
            f"{r['matrix']}: negative values clipped",
            r.get("n_negative_cells_clipped", ""),
            "Technical negative abundance values were clipped to zero before log1p transformation."
        )
        add_metric(
            metrics,
            "VOC trait matrix",
            f"{r['matrix']}: final NaN",
            r.get("n_nan_final", ""),
            "Final VOC trait matrix should contain no missing values."
        )

wgcna_input = read_tsv(project / "analysis/wgcna_inputs/wgcna_input_preparation_summary.tsv")
if not wgcna_input.empty:
    for _, r in wgcna_input.iterrows():
        add_metric(
            metrics,
            "WGCNA input",
            f"{r['organ']}: RNA-seq samples",
            r.get("n_samples", ""),
            "Sample-level expression matrix used for WGCNA module construction."
        )
        add_metric(
            metrics,
            "WGCNA input",
            f"{r['organ']}: selected variable genes",
            r.get("n_selected_genes", ""),
            "Top variable VST genes used for co-expression network construction."
        )

wgcna_run = read_tsv(project / "analysis/wgcna_modules/wgcna_run_summary.tsv")
if not wgcna_run.empty:
    for _, r in wgcna_run.iterrows():
        add_metric(
            metrics,
            "WGCNA modules",
            f"{r['organ']}: selected soft power",
            r.get("selected_power", ""),
            "Soft-thresholding power selected by scale-free topology diagnostics."
        )
        add_metric(
            metrics,
            "WGCNA modules",
            f"{r['organ']}: non-grey modules",
            r.get("n_modules_non_grey", ""),
            "Detected co-expression modules excluding grey/unassigned genes."
        )

assoc_summary = read_tsv(project / "analysis/module_trait_association/module_trait_association_summary.tsv")
if not assoc_summary.empty and {"item", "value"}.issubset(assoc_summary.columns):
    assoc = dict(zip(assoc_summary["item"], assoc_summary["value"]))
    for k, v in assoc.items():
        add_metric(
            metrics,
            "Module-trait association",
            k,
            v,
            "Module-module and module-VOC association statistics."
        )

enrich_summary = read_tsv(project / "analysis/module_gene_enrichment/module_gene_enrichment_summary.tsv")
if not enrich_summary.empty and {"item", "value"}.issubset(enrich_summary.columns):
    enr = dict(zip(enrich_summary["item"], enrich_summary["value"]))
    for k, v in enr.items():
        add_metric(
            metrics,
            "Module enrichment and hubs",
            k,
            v,
            "Gene-set enrichment and VOC-related hub gene summary."
        )

trip_summary = read_tsv(project / "analysis/candidate_gene_triplets_annotated/candidate_triplet_annotation_summary.tsv")
if not trip_summary.empty and {"item", "value"}.issubset(trip_summary.columns):
    ts = dict(zip(trip_summary["item"], trip_summary["value"]))
    for k, v in ts.items():
        add_metric(
            metrics,
            "Candidate gene triplets",
            k,
            v,
            "Candidate leaf hub gene-peel hub gene-peel VOC association triplets."
        )

metrics_df = pd.DataFrame(metrics)
write_tsv(metrics_df, "final_evidence_key_metrics.tsv")

# ------------------------------------------------------------------
# 2. Top result tables
# ------------------------------------------------------------------

mm = read_tsv(project / "analysis/module_trait_association/leaf_module_vs_peel_module_spearman_overall.tsv")
mm_top = add_abs_sort(mm).head(50)
write_tsv(mm_top, "top_leaf_peel_module_pairs.tsv")

mm_resid = read_tsv(project / "analysis/module_trait_association/leaf_module_vs_peel_module_stage_residual_pearson.tsv")
mm_resid_top = add_abs_sort(mm_resid).head(50)
write_tsv(mm_resid_top, "top_leaf_peel_module_pairs_stage_residual.tsv")

mv = read_tsv(project / "analysis/module_trait_association/module_vs_peel_key_voc_spearman_overall.tsv")
mv = annotate_trait(mv, "right")
mv_top = add_abs_sort(mv).head(80)
write_tsv(mv_top, "top_module_peel_voc_associations.tsv")

mv_resid = read_tsv(project / "analysis/module_trait_association/module_vs_peel_key_voc_stage_residual_pearson.tsv")
mv_resid = annotate_trait(mv_resid, "right")
mv_resid_top = add_abs_sort(mv_resid).head(80)
write_tsv(mv_resid_top, "top_module_peel_voc_associations_stage_residual.tsv")

chains = read_tsv(project / "analysis/module_trait_association/candidate_leaf_peel_module_voc_chains_top.tsv")
if not chains.empty:
    chains = chains.sort_values("evidence_score", ascending=False, na_position="last")
chains_top = chains.head(100)
write_tsv(chains_top, "top_candidate_module_voc_chains.tsv")

manuscript_trip = read_tsv(project / "analysis/candidate_gene_triplets_annotated/candidate_triplets_manuscript_examples_compact.tsv")
write_tsv(manuscript_trip, "candidate_triplets_manuscript_examples.tsv")

reviewer_trip = read_tsv(project / "analysis/candidate_gene_triplets_annotated/candidate_triplets_reviewer_ready_compact.tsv")
write_tsv(reviewer_trip, "candidate_triplets_reviewer_ready.tsv")

gene_freq = read_tsv(project / "analysis/candidate_gene_triplets_annotated/candidate_gene_frequency_summary.tsv")
if not gene_freq.empty:
    gene_freq["n_vocs"] = pd.to_numeric(gene_freq.get("n_vocs"), errors="coerce")
    gene_freq["n_triplets"] = pd.to_numeric(gene_freq.get("n_triplets"), errors="coerce")
    gene_freq["max_evidence_score"] = pd.to_numeric(gene_freq.get("max_evidence_score"), errors="coerce")
    gene_freq = gene_freq.sort_values(
        ["organ", "n_vocs", "n_triplets", "max_evidence_score"],
        ascending=[True, False, False, False],
        na_position="last"
    )
gene_freq_top = gene_freq.groupby("organ", group_keys=False).head(50) if not gene_freq.empty and "organ" in gene_freq.columns else gene_freq.head(100)
write_tsv(gene_freq_top, "candidate_gene_frequency_summary_top.tsv")

key_mod = read_tsv(project / "analysis/module_gene_enrichment/key_module_summary_with_enrichment_and_hubs.tsv")
write_tsv(key_mod, "key_module_summary_with_hubs.tsv")

tier_summary = read_tsv(project / "analysis/candidate_gene_triplets_annotated/candidate_triplet_tier_summary.tsv")
voc_tier_summary = read_tsv(project / "analysis/candidate_gene_triplets_annotated/candidate_triplet_per_voc_tier_summary.tsv")
write_tsv(tier_summary, "candidate_triplet_tier_summary.tsv")
write_tsv(voc_tier_summary, "candidate_triplet_per_voc_tier_summary.tsv")

# ------------------------------------------------------------------
# 3. Markdown evidence summary
# ------------------------------------------------------------------

def metric_value(metric_name):
    if metrics_df.empty:
        return ""
    hit = metrics_df[metrics_df["metric"] == metric_name]
    if hit.empty:
        return ""
    return str(hit.iloc[0]["value"])

# Extract some headline values by more flexible lookup
def metric_contains(substr):
    if metrics_df.empty:
        return ""
    hit = metrics_df[metrics_df["metric"].astype(str).str.contains(substr, regex=False)]
    if hit.empty:
        return ""
    return str(hit.iloc[0]["value"])

md = []
md.append("# Transcriptome–VOC Evidence Package Summary\n")
md.append("## 1. Purpose\n")
md.append(
    "This package summarizes the additional transcriptomic analyses used to provide biological and molecular context for the leaf–peel VOC proxy framework. "
    "The analyses are intended to support biological plausibility and candidate marker interpretation, not to establish direct regulatory causality.\n"
)

md.append("## 2. Evidence chain\n")
md.append(
    "The current evidence chain is:\n\n"
    "1. Matched leaf and peel RNA-seq profiles show global cross-organ transcriptomic coordination.\n"
    "2. VOC/metabolism-related gene sets retain supportive cross-organ coordination, especially at the mature stage.\n"
    "3. Leaf-only and peel-only WGCNA identified co-expression modules from 208 RNA-seq samples per organ.\n"
    "4. Leaf and peel module eigengenes were averaged to 52 cultivar-stage profiles and correlated with matched VOC traits.\n"
    "5. Cross-organ module pairs and module–peel VOC associations were used to derive candidate module–VOC chains.\n"
    "6. Representative module chains were refined into candidate leaf hub gene–peel hub gene–peel VOC triplets.\n"
)

md.append("## 3. Key metrics\n")
md.append(df_to_md(metrics_df, max_rows=80))

md.append("## 4. Top leaf module–peel module associations\n")
md.append(df_to_md(mm_top, max_rows=15))

md.append("## 5. Top stage-residual leaf module–peel module associations\n")
md.append(df_to_md(mm_resid_top, max_rows=15))

md.append("## 6. Top module–peel VOC associations\n")
md.append(df_to_md(mv_top, max_rows=20))

md.append("## 7. Top candidate module–VOC chains\n")
md.append(df_to_md(chains_top, max_rows=20))

md.append("## 8. Candidate gene triplet tier summary\n")
md.append(df_to_md(tier_summary, max_rows=20))

md.append("## 9. Candidate gene triplets by VOC\n")
md.append(df_to_md(voc_tier_summary, max_rows=50))

md.append("## 10. Manuscript-level candidate gene examples\n")
md.append(df_to_md(manuscript_trip, max_rows=30))

md.append("## 11. Interpretation boundary\n")
md.append(
    "Recommended interpretation:\n\n"
    "- These results provide association-level molecular support for the leaf–peel VOC proxy framework.\n"
    "- The strongest support is at the module and candidate marker level, especially for decanal-related chains.\n"
    "- The results should be described as cross-organ coordinated modules, VOC-associated modules, candidate module–VOC chains, and candidate gene triplets.\n"
    "- The results should not be described as evidence that leaf genes directly regulate peel VOC biosynthesis or that specific leaf genes causally determine specific peel genes.\n"
)

md.append("## 12. Recommended manuscript wording\n")
md.append(
    "Possible wording:\n\n"
    "> Matched leaf and peel transcriptomes showed significant cross-organ coordination at both global and module levels. "
    "Co-expression network analysis further identified leaf and peel modules associated with key peel VOC traits. "
    "Within representative module–VOC chains, we identified candidate leaf–peel hub gene pairs whose expression patterns were correlated with each other and with corresponding peel VOCs, especially decanal-related traits. "
    "These associations provide candidate molecular support for the leaf–peel volatile proxy relationship, but they do not establish direct regulatory causality.\n"
)

md_path = out_dir / "transcriptome_voc_evidence_package_summary.md"
md_path.write_text("\n".join(md), encoding="utf-8")
print(f"Wrote {md_path}")

# ------------------------------------------------------------------
# 4. Proposed supplementary figures and tables
# ------------------------------------------------------------------

fig_md = []
fig_md.append("# Proposed Supplementary Figures and Tables\n")

fig_md.append("## Supplementary Figure X. Cross-organ transcriptomic coordination between leaf and peel\n")
fig_md.append(
    "Recommended panels:\n\n"
    "A. RNA-seq PCA of all 416 samples.\n"
    "B. PERMANOVA R2 contributions for organ, cultivar, stage, and organ-by-stage effects.\n"
    "C. Global leaf–peel transcriptome distance correlation across 52 cultivar-stage profiles.\n"
    "D. Stage-specific global Mantel r values.\n"
    "E. VOC/metabolism gene-set-level Mantel r values with random gene-set background.\n"
    "F. Stage-specific VOC/metabolism gene-set heatmap.\n"
)

fig_md.append("## Supplementary Figure Y. Module-level and VOC-linked cross-organ coordination\n")
fig_md.append(
    "Recommended panels:\n\n"
    "A. Leaf WGCNA module dendrogram.\n"
    "B. Peel WGCNA module dendrogram.\n"
    "C. Leaf module–peel module correlation heatmap.\n"
    "D. Leaf/peel modules vs key peel VOC heatmap.\n"
    "E. Candidate module–VOC chain network, highlighting representative Decanal and (E)-β-Ocimene chains.\n"
    "F. Representative candidate leaf hub gene–peel hub gene–peel VOC triplets.\n"
)

fig_md.append("## Supplementary Tables\n")
fig_md.append(
    "Recommended supplementary tables:\n\n"
    "1. RNA-seq sample metadata and quality summary.\n"
    "2. VOC trait matrix preparation summary.\n"
    "3. Leaf and peel WGCNA module sizes and module membership.\n"
    "4. Leaf module–peel module correlation table.\n"
    "5. Module–key VOC correlation table.\n"
    "6. Candidate module–VOC chains.\n"
    "7. Candidate leaf hub gene–peel hub gene–peel VOC triplets with annotations.\n"
)

fig_md.append("## Main-text recommendation\n")
fig_md.append(
    "The transcriptomic results should be summarized in the main text in a compact paragraph and placed mainly in Supplementary Figures/Tables. "
    "This keeps the manuscript focused on the leaf–peel VOC proxy framework while addressing the reviewer’s concern about biological support.\n"
)

fig_path = out_dir / "proposed_supplementary_figures_and_tables.md"
fig_path.write_text("\n".join(fig_md), encoding="utf-8")
print(f"Wrote {fig_path}")

# ------------------------------------------------------------------
# 5. Copy selected existing high-value tables into final package
# ------------------------------------------------------------------

copy_files = {
    "source_candidate_triplets_top_annotated.tsv": project / "analysis/candidate_gene_triplets_annotated/candidate_triplets_top_annotated.tsv",
    "source_module_trait_association_summary.tsv": project / "analysis/module_trait_association/module_trait_association_summary.tsv",
    "source_module_gene_enrichment_summary.tsv": project / "analysis/module_gene_enrichment/module_gene_enrichment_summary.tsv",
    "source_wgcna_run_summary.tsv": project / "analysis/wgcna_modules/wgcna_run_summary.tsv",
}

for dest, src in copy_files.items():
    if src.exists():
        shutil.copy2(src, out_dir / dest)
        print(f"Copied {src} -> {out_dir / dest}")

print("\nFinal evidence package complete.")
