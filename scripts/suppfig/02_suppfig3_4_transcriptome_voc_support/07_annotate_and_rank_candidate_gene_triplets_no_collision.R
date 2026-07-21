suppressPackageStartupMessages({
  library(data.table)
})

options(stringsAsFactors = FALSE)

project <- Sys.getenv("PROJECT_DIR", unset = ".")
project <- normalizePath(project, mustWork = FALSE)

triplet_dir <- file.path(project, "analysis/candidate_gene_triplets")
enrich_dir <- file.path(project, "analysis/module_gene_enrichment")
out_dir <- file.path(project, "analysis/candidate_gene_triplets_annotated")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

triplet_top_file <- file.path(triplet_dir, "candidate_leaf_gene_peel_gene_peel_voc_triplets_top.tsv")
hub_annot_file <- file.path(enrich_dir, "key_module_voc_related_hub_genes.tsv")
module_summary_file <- file.path(enrich_dir, "key_module_summary_with_enrichment_and_hubs.tsv")

tier_order <- c(
  "Tier 1: overall strong and stage-residual robust",
  "Tier 2: overall strong and partly stage-residual robust",
  "Tier 3: overall candidate"
)

truncate_text <- function(x, n = 180) {
  x <- as.character(x)
  x[is.na(x)] <- ""
  ifelse(nchar(x) > n, paste0(substr(x, 1, n), "..."), x)
}

left_join_safe <- function(x, y, by_col, label) {
  x <- as.data.table(x)
  y <- as.data.table(y)

  if (!by_col %in% names(x)) {
    stop(label, ": ", by_col, " not in x. x columns: ", paste(names(x), collapse = ", "))
  }
  if (!by_col %in% names(y)) {
    stop(label, ": ", by_col, " not in y. y columns: ", paste(names(y), collapse = ", "))
  }

  merge(x, y, by = by_col, all.x = TRUE, sort = FALSE)
}

make_gene_annot <- function(hub, organ_label, gene_key, prefix) {
  annot <- copy(hub[organ == organ_label])

  if (nrow(annot) == 0) {
    out <- data.table(tmp = character())
    setnames(out, "tmp", gene_key)
    return(out)
  }

  if (!"Geneid" %in% names(annot)) {
    stop("Hub file missing Geneid column.")
  }

  if (!"abs_kME_own_module" %in% names(annot)) {
    if ("kME_own_module" %in% names(annot)) {
      annot[, abs_kME_own_module := abs(as.numeric(kME_own_module))]
    } else {
      annot[, abs_kME_own_module := NA_real_]
    }
  }

  # 关键修正：不保留 module / module_id，避免和 triplet 主表中的 leaf_module / peel_module 冲突
  keep_cols <- intersect(
    c(
      "Geneid",
      "kME_own_module",
      "abs_kME_own_module",
      "VOC_gene_sets",
      "KO",
      "Pathway_direct",
      "Pathway_from_KO",
      "Pathway_all",
      "GO",
      "InterPro",
      "in_filtered_vst"
    ),
    names(annot)
  )

  annot <- annot[, ..keep_cols]
  annot <- unique(annot, by = "Geneid")

  setnames(annot, "Geneid", gene_key)

  other_cols <- setdiff(names(annot), gene_key)
  setnames(annot, other_cols, paste0(prefix, other_cols))

  annot
}

make_module_map <- function(module_summary, organ_label, module_key, prefix) {
  ms <- copy(module_summary[organ == organ_label])

  if (nrow(ms) == 0) {
    out <- data.table(tmp = character())
    setnames(out, "tmp", module_key)
    return(out)
  }

  if (!all(c("organ", "module") %in% names(ms))) {
    stop("Module summary missing organ/module columns.")
  }

  ms[, module_full := paste0(ifelse(organ == "Leaf", "LeafME__", "PeelME__"), module)]

  keep_cols <- intersect(
    c(
      "module_full",
      "evidence_sources",
      "N",
      "n_voc_related_genes",
      "top_enriched_gene_sets",
      "n_significant_enriched_gene_sets",
      "top_voc_related_hub_genes",
      "top_voc_related_hub_kME"
    ),
    names(ms)
  )

  ms <- ms[, ..keep_cols]
  ms <- unique(ms, by = "module_full")

  setnames(ms, "module_full", module_key)

  other_cols <- setdiff(names(ms), module_key)
  setnames(ms, other_cols, paste0(prefix, other_cols))

  ms
}

score_tiers <- function(dt) {
  numeric_cols <- intersect(
    c(
      "leaf_gene_peel_gene_stage_resid_cor",
      "leaf_gene_peel_gene_stage_resid_p",
      "leaf_gene_peel_voc_stage_resid_cor",
      "leaf_gene_peel_voc_stage_resid_p",
      "peel_gene_peel_voc_stage_resid_cor",
      "peel_gene_peel_voc_stage_resid_p",
      "leaf_gene_peel_gene_fdr",
      "leaf_gene_peel_voc_fdr",
      "peel_gene_peel_voc_fdr",
      "leaf_gene_peel_gene_cor",
      "leaf_gene_peel_voc_cor",
      "peel_gene_peel_voc_cor",
      "evidence_score"
    ),
    names(dt)
  )

  for (cc in numeric_cols) {
    dt[[cc]] <- as.numeric(dt[[cc]])
  }

  if (!is.logical(dt$direction_gene_coherent)) {
    dt[, direction_gene_coherent := as.logical(direction_gene_coherent)]
  }
  if (!is.logical(dt$module_direction_consistent)) {
    dt[, module_direction_consistent := as.logical(module_direction_consistent)]
  }

  dt[, stage_pass_leaf_peel_gene :=
       abs(leaf_gene_peel_gene_stage_resid_cor) >= 0.40 &
       leaf_gene_peel_gene_stage_resid_p < 0.05]

  dt[, stage_pass_leaf_gene_voc :=
       abs(leaf_gene_peel_voc_stage_resid_cor) >= 0.40 &
       leaf_gene_peel_voc_stage_resid_p < 0.05]

  dt[, stage_pass_peel_gene_voc :=
       abs(peel_gene_peel_voc_stage_resid_cor) >= 0.40 &
       peel_gene_peel_voc_stage_resid_p < 0.05]

  dt[, n_stage_residual_pass := rowSums(
    cbind(
      stage_pass_leaf_peel_gene,
      stage_pass_leaf_gene_voc,
      stage_pass_peel_gene_voc
    ),
    na.rm = TRUE
  )]

  dt[, all_overall_fdr005 :=
       leaf_gene_peel_gene_fdr < 0.05 &
       leaf_gene_peel_voc_fdr < 0.05 &
       peel_gene_peel_voc_fdr < 0.05]

  dt[, overall_strong :=
       abs(leaf_gene_peel_gene_cor) >= 0.50 &
       abs(leaf_gene_peel_voc_cor) >= 0.45 &
       abs(peel_gene_peel_voc_cor) >= 0.45]

  dt[, evidence_tier := fifelse(
    direction_gene_coherent == TRUE &
      module_direction_consistent == TRUE &
      all_overall_fdr005 == TRUE &
      overall_strong == TRUE &
      n_stage_residual_pass == 3,
    "Tier 1: overall strong and stage-residual robust",
    fifelse(
      direction_gene_coherent == TRUE &
        module_direction_consistent == TRUE &
        all_overall_fdr005 == TRUE &
        overall_strong == TRUE &
        n_stage_residual_pass >= 2,
      "Tier 2: overall strong and partly stage-residual robust",
      "Tier 3: overall candidate"
    )
  )]

  dt[, evidence_tier_rank := match(evidence_tier, tier_order)]

  dt
}

make_compact <- function(dt) {
  keep_cols <- c(
    "evidence_tier",
    "peel_voc_original",
    "leaf_module",
    "peel_module",
    "leaf_gene",
    "peel_gene",

    "leaf_gene_peel_gene_cor",
    "leaf_gene_peel_gene_stage_resid_cor",
    "leaf_gene_peel_voc_cor",
    "leaf_gene_peel_voc_stage_resid_cor",
    "peel_gene_peel_voc_cor",
    "peel_gene_peel_voc_stage_resid_cor",
    "evidence_score",

    "leaf_kME_own_module",
    "peel_kME_own_module",
    "leaf_abs_kME_own_module",
    "peel_abs_kME_own_module",

    "leaf_VOC_gene_sets",
    "peel_VOC_gene_sets",
    "leaf_KO",
    "peel_KO",
    "leaf_Pathway_all",
    "peel_Pathway_all",
    "leaf_GO",
    "peel_GO",
    "leaf_InterPro",
    "peel_InterPro",

    "leaf_module_evidence_sources",
    "peel_module_evidence_sources",
    "leaf_module_top_enriched_gene_sets",
    "peel_module_top_enriched_gene_sets",
    "leaf_module_top_voc_related_hub_genes",
    "peel_module_top_voc_related_hub_genes"
  )

  keep_cols <- intersect(keep_cols, names(dt))
  out <- copy(dt[, ..keep_cols])

  text_cols <- grep("gene_sets|KO|Pathway|GO|InterPro|hub_genes|evidence_sources", names(out), value = TRUE)
  for (cc in text_cols) {
    out[[cc]] <- truncate_text(out[[cc]], 180)
  }

  out
}

# -----------------------------
# Read inputs
# -----------------------------

if (!file.exists(triplet_top_file)) stop("Missing file: ", triplet_top_file)
if (!file.exists(hub_annot_file)) stop("Missing file: ", hub_annot_file)

trip <- fread(triplet_top_file)
hub <- fread(hub_annot_file)

message("Input triplets: ", nrow(trip))
message("Input hubs: ", nrow(hub))

required <- c(
  "leaf_module", "peel_module", "leaf_gene", "peel_gene",
  "peel_voc_original", "direction_gene_coherent",
  "module_direction_consistent", "evidence_score"
)
missing_required <- setdiff(required, names(trip))
if (length(missing_required) > 0) {
  stop("Triplet file missing columns: ", paste(missing_required, collapse = ", "))
}

trip <- score_tiers(trip)

leaf_gene_annot <- make_gene_annot(hub, "Leaf", "leaf_gene", "leaf_")
peel_gene_annot <- make_gene_annot(hub, "Peel", "peel_gene", "peel_")

trip <- left_join_safe(trip, leaf_gene_annot, "leaf_gene", "leaf gene annotation")
trip <- left_join_safe(trip, peel_gene_annot, "peel_gene", "peel gene annotation")

if (file.exists(module_summary_file)) {
  ms <- fread(module_summary_file)

  leaf_module_map <- make_module_map(ms, "Leaf", "leaf_module", "leaf_module_")
  peel_module_map <- make_module_map(ms, "Peel", "peel_module", "peel_module_")

  trip <- left_join_safe(trip, leaf_module_map, "leaf_module", "leaf module annotation")
  trip <- left_join_safe(trip, peel_module_map, "peel_module", "peel module annotation")
}

# Final order
setorder(trip, evidence_tier_rank, peel_voc_original, -evidence_score)

# Full annotated output
fwrite(trip, file.path(out_dir, "candidate_triplets_top_annotated.tsv"), sep = "\t")

# Summaries
tier_summary <- trip[, .N, by = evidence_tier][order(match(evidence_tier, tier_order))]
fwrite(tier_summary, file.path(out_dir, "candidate_triplet_tier_summary.tsv"), sep = "\t")

voc_tier_summary <- trip[
  ,
  .N,
  by = .(peel_voc_original, evidence_tier)
][order(peel_voc_original, match(evidence_tier, tier_order))]

fwrite(voc_tier_summary, file.path(out_dir, "candidate_triplet_per_voc_tier_summary.tsv"), sep = "\t")

# Reviewer-ready compact table: top 8 per VOC
reviewer_ready <- trip[
  ,
  head(.SD[order(evidence_tier_rank, -evidence_score)], 8),
  by = .(peel_voc_original)
]
reviewer_ready_compact <- make_compact(reviewer_ready)

fwrite(
  reviewer_ready_compact,
  file.path(out_dir, "candidate_triplets_reviewer_ready_compact.tsv"),
  sep = "\t"
)

# Manuscript examples: Tier 1/2 only, top 5 per VOC
manuscript_examples <- trip[
  evidence_tier %in% c(
    "Tier 1: overall strong and stage-residual robust",
    "Tier 2: overall strong and partly stage-residual robust"
  )
]

manuscript_examples <- manuscript_examples[
  ,
  head(.SD[order(-evidence_score)], 5),
  by = .(peel_voc_original)
]

manuscript_examples_compact <- make_compact(manuscript_examples)

fwrite(
  manuscript_examples_compact,
  file.path(out_dir, "candidate_triplets_manuscript_examples_compact.tsv"),
  sep = "\t"
)

# Gene frequency
gene_freq_leaf <- trip[
  ,
  .(
    organ = "Leaf",
    n_triplets = .N,
    n_vocs = length(unique(peel_voc_original)),
    vocs = paste(sort(unique(peel_voc_original)), collapse = ";"),
    max_evidence_score = max(evidence_score, na.rm = TRUE),
    best_tier = tier_order[min(evidence_tier_rank, na.rm = TRUE)]
  ),
  by = .(gene = leaf_gene)
]

gene_freq_peel <- trip[
  ,
  .(
    organ = "Peel",
    n_triplets = .N,
    n_vocs = length(unique(peel_voc_original)),
    vocs = paste(sort(unique(peel_voc_original)), collapse = ";"),
    max_evidence_score = max(evidence_score, na.rm = TRUE),
    best_tier = tier_order[min(evidence_tier_rank, na.rm = TRUE)]
  ),
  by = .(gene = peel_gene)
]

gene_freq <- rbindlist(list(gene_freq_leaf, gene_freq_peel), fill = TRUE)
setorder(gene_freq, organ, -n_vocs, -n_triplets, -max_evidence_score)

fwrite(gene_freq, file.path(out_dir, "candidate_gene_frequency_summary.tsv"), sep = "\t")

summary <- data.table(
  item = c(
    "n_top_triplets",
    "n_vocs",
    "n_tier1_triplets",
    "n_tier2_triplets",
    "n_tier3_triplets",
    "n_unique_leaf_genes",
    "n_unique_peel_genes",
    "n_annotated_leaf_genes",
    "n_annotated_peel_genes"
  ),
  value = c(
    nrow(trip),
    length(unique(trip$peel_voc_original)),
    nrow(trip[evidence_tier == "Tier 1: overall strong and stage-residual robust"]),
    nrow(trip[evidence_tier == "Tier 2: overall strong and partly stage-residual robust"]),
    nrow(trip[evidence_tier == "Tier 3: overall candidate"]),
    length(unique(trip$leaf_gene)),
    length(unique(trip$peel_gene)),
    length(unique(trip[!is.na(leaf_VOC_gene_sets) & leaf_VOC_gene_sets != ""]$leaf_gene)),
    length(unique(trip[!is.na(peel_VOC_gene_sets) & peel_VOC_gene_sets != ""]$peel_gene))
  )
)

fwrite(summary, file.path(out_dir, "candidate_triplet_annotation_summary.tsv"), sep = "\t")

message("\nSummary:")
print(summary)

message("\nTier summary:")
print(tier_summary)

message("\nPer-VOC tier summary:")
print(voc_tier_summary)

message("\nReviewer-ready compact preview:")
print(head(reviewer_ready_compact, 30))

message("\nDone.")
