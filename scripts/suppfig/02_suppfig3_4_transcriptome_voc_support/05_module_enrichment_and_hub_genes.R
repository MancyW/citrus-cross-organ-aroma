suppressPackageStartupMessages({
  library(data.table)
})

options(stringsAsFactors = FALSE)

project <- Sys.getenv("PROJECT_DIR", unset = ".")
project <- normalizePath(project, mustWork = FALSE)

wgcna_dir <- file.path(project, "analysis/wgcna_modules")
assoc_dir <- file.path(project, "analysis/module_trait_association")
gene_set_file <- file.path(project, "analysis/gene_sets/VOC_related_gene_sets.filtered_long.tsv")
gene_function_file <- file.path(project, "annotation/summary/gene_function_summary.tsv")

out_dir <- file.path(project, "analysis/module_gene_enrichment")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

leaf_membership_file <- file.path(wgcna_dir, "leaf/leaf_gene_module_membership.tsv")
peel_membership_file <- file.path(wgcna_dir, "peel/peel_gene_module_membership.tsv")

candidate_top_file <- file.path(assoc_dir, "candidate_leaf_peel_module_voc_chains_top.tsv")
candidate_strict_file <- file.path(assoc_dir, "candidate_leaf_peel_module_voc_chains_top_strict.tsv")
module_peel_voc_file <- file.path(assoc_dir, "module_vs_peel_key_voc_spearman_overall.tsv")
module_module_file <- file.path(assoc_dir, "leaf_module_vs_peel_module_spearman_overall.tsv")

detect_col <- function(dt, candidates, pattern = NULL) {
  for (cc in candidates) {
    if (cc %in% colnames(dt)) return(cc)
  }
  if (!is.null(pattern)) {
    hits <- grep(pattern, colnames(dt), ignore.case = TRUE, value = TRUE)
    if (length(hits) > 0) return(hits[1])
  }
  stop("Could not detect required column. Available columns: ", paste(colnames(dt), collapse = ", "))
}

clean_module_name <- function(x) {
  x <- gsub("^LeafME__", "", x)
  x <- gsub("^PeelME__", "", x)
  x
}

module_id <- function(organ, module) {
  paste0(organ, "__", module)
}

read_membership <- function(path, organ) {
  dt <- fread(path)
  if (!all(c("Geneid", "module", "kME_own_module") %in% colnames(dt))) {
    stop("Membership file missing required columns: ", path)
  }
  dt[, organ := organ]
  dt[, module_id := module_id(organ, module)]
  dt
}

leaf_mem <- read_membership(leaf_membership_file, "Leaf")
peel_mem <- read_membership(peel_membership_file, "Peel")
all_mem <- rbindlist(list(leaf_mem, peel_mem), fill = TRUE)

# ------------------------------------------------------------------
# 1. Select key modules from association results
# ------------------------------------------------------------------

key_modules <- data.table(
  organ = character(),
  module = character(),
  module_id = character(),
  evidence_source = character()
)

if (file.exists(candidate_top_file)) {
  cand_top <- fread(candidate_top_file)
  if (nrow(cand_top) > 0) {
    leaf_mods <- unique(clean_module_name(cand_top$leaf_module))
    peel_mods <- unique(clean_module_name(cand_top$peel_module))
    key_modules <- rbind(
      key_modules,
      data.table(organ = "Leaf", module = leaf_mods,
                 module_id = module_id("Leaf", leaf_mods),
                 evidence_source = "candidate_chain_top"),
      data.table(organ = "Peel", module = peel_mods,
                 module_id = module_id("Peel", peel_mods),
                 evidence_source = "candidate_chain_top"),
      fill = TRUE
    )
  }
}

if (file.exists(candidate_strict_file)) {
  cand_strict <- fread(candidate_strict_file)
  if (nrow(cand_strict) > 0) {
    leaf_mods <- unique(clean_module_name(cand_strict$leaf_module))
    peel_mods <- unique(clean_module_name(cand_strict$peel_module))
    key_modules <- rbind(
      key_modules,
      data.table(organ = "Leaf", module = leaf_mods,
                 module_id = module_id("Leaf", leaf_mods),
                 evidence_source = "candidate_chain_strict"),
      data.table(organ = "Peel", module = peel_mods,
                 module_id = module_id("Peel", peel_mods),
                 evidence_source = "candidate_chain_strict"),
      fill = TRUE
    )
  }
}

# Add modules with strong direct module-peel VOC association
if (file.exists(module_peel_voc_file)) {
  mv <- fread(module_peel_voc_file)
  mv_sig <- mv[padj_BH < 0.05 & abs(cor) >= 0.5]
  if (nrow(mv_sig) > 0) {
    mod_organ <- ifelse(grepl("^LeafME__", mv_sig$left), "Leaf", "Peel")
    mod_color <- clean_module_name(mv_sig$left)
    key_modules <- rbind(
      key_modules,
      data.table(
        organ = mod_organ,
        module = mod_color,
        module_id = module_id(mod_organ, mod_color),
        evidence_source = "direct_module_peel_voc"
      ),
      fill = TRUE
    )
  }
}

# Add modules from strong cross-organ pairs
if (file.exists(module_module_file)) {
  mm <- fread(module_module_file)
  mm_sig <- mm[padj_BH < 0.05 & abs(cor) >= 0.5]
  if (nrow(mm_sig) > 0) {
    leaf_mods <- unique(clean_module_name(mm_sig$left))
    peel_mods <- unique(clean_module_name(mm_sig$right))
    key_modules <- rbind(
      key_modules,
      data.table(organ = "Leaf", module = leaf_mods,
                 module_id = module_id("Leaf", leaf_mods),
                 evidence_source = "cross_organ_module_pair"),
      data.table(organ = "Peel", module = peel_mods,
                 module_id = module_id("Peel", peel_mods),
                 evidence_source = "cross_organ_module_pair"),
      fill = TRUE
    )
  }
}

key_modules <- unique(key_modules)

# Collapse evidence source by module
key_module_summary <- key_modules[
  ,
  .(evidence_sources = paste(sort(unique(evidence_source)), collapse = ";")),
  by = .(organ, module, module_id)
]

# Remove grey if present
key_module_summary <- key_module_summary[module != "grey"]

fwrite(key_module_summary, file.path(out_dir, "selected_key_modules.tsv"), sep = "\t")

message("Selected key modules:")
print(key_module_summary)

# ------------------------------------------------------------------
# 2. Read VOC-related gene sets
# ------------------------------------------------------------------

gs <- fread(gene_set_file)

gene_col <- detect_col(gs, c("Geneid", "gene", "gene_id", "GeneID"), pattern = "gene")
set_col <- detect_col(gs, c("gene_set", "GeneSet", "set", "term", "category"), pattern = "set|term|category")

gs <- gs[, .(
  Geneid = as.character(get(gene_col)),
  gene_set = as.character(get(set_col))
)]

gs <- unique(gs[!is.na(Geneid) & Geneid != "" & !is.na(gene_set) & gene_set != ""])

message("Gene sets loaded: ", length(unique(gs$gene_set)))
message("Gene set genes loaded: ", length(unique(gs$Geneid)))

# Gene function summary
if (file.exists(gene_function_file)) {
  gf <- fread(gene_function_file)
  if (!"Geneid" %in% colnames(gf)) {
    stop("gene_function_summary.tsv does not contain Geneid column.")
  }
} else {
  gf <- data.table(Geneid = unique(all_mem$Geneid))
}

# Add gene-set membership string for hub gene annotation
gs_str <- gs[, .(VOC_gene_sets = paste(sort(unique(gene_set)), collapse = ";")), by = Geneid]

# ------------------------------------------------------------------
# 3. Hypergeometric enrichment for selected modules
# ------------------------------------------------------------------

enrich_rows <- list()
row_i <- 1

for (org in c("Leaf", "Peel")) {
  mem_org <- all_mem[organ == org]
  universe <- unique(mem_org$Geneid)

  gs_org <- gs[Geneid %in% universe]
  set_list <- split(gs_org$Geneid, gs_org$gene_set)

  selected_mods <- key_module_summary[organ == org]$module

  for (mod in selected_mods) {
    module_genes <- unique(mem_org[module == mod]$Geneid)
    N <- length(module_genes)
    M <- length(universe)

    for (set_name in names(set_list)) {
      set_genes <- unique(set_list[[set_name]])
      K <- length(set_genes)
      x <- length(intersect(module_genes, set_genes))

      if (K == 0 || N == 0) next

      p <- phyper(q = x - 1, m = K, n = M - K, k = N, lower.tail = FALSE)
      fold_enrichment <- (x / N) / (K / M)

      enrich_rows[[row_i]] <- data.table(
        organ = org,
        module = mod,
        module_id = module_id(org, mod),
        gene_set = set_name,
        universe_genes = M,
        module_genes = N,
        gene_set_genes_in_universe = K,
        overlap_genes = x,
        fold_enrichment = fold_enrichment,
        p_value = p,
        overlap_gene_ids = paste(sort(intersect(module_genes, set_genes)), collapse = ";")
      )
      row_i <- row_i + 1
    }
  }
}

enrich <- rbindlist(enrich_rows, fill = TRUE)
enrich[, padj_BH := p.adjust(p_value, method = "BH")]
setorder(enrich, padj_BH, -fold_enrichment, -overlap_genes)

fwrite(enrich, file.path(out_dir, "key_module_voc_gene_set_enrichment.tsv"), sep = "\t")

enrich_sig <- enrich[padj_BH < 0.05 & overlap_genes >= 3]
fwrite(enrich_sig, file.path(out_dir, "key_module_voc_gene_set_enrichment_significant.tsv"), sep = "\t")

message("Enrichment tests: ", nrow(enrich))
message("Significant enrichments FDR<0.05 overlap>=3: ", nrow(enrich_sig))
print(head(enrich_sig, 30))

# ------------------------------------------------------------------
# 4. Hub genes in key modules
# ------------------------------------------------------------------

hub_all <- all_mem[module_id %in% key_module_summary$module_id & module != "grey"]

# Add gene set and function annotation
hub_all <- merge(hub_all, gs_str, by = "Geneid", all.x = TRUE)
hub_all <- merge(hub_all, gf, by = "Geneid", all.x = TRUE)

hub_all[, abs_kME_own_module := abs(as.numeric(kME_own_module))]
setorder(hub_all, organ, module, -abs_kME_own_module)

fwrite(hub_all, file.path(out_dir, "key_module_all_genes_with_annotation.tsv"), sep = "\t")

hub_top50 <- hub_all[
  ,
  head(.SD, 50),
  by = .(organ, module, module_id)
]

fwrite(hub_top50, file.path(out_dir, "key_module_top50_hub_genes.tsv"), sep = "\t")

# More focused: top VOC-related hub genes
hub_voc_related <- hub_all[!is.na(VOC_gene_sets) & VOC_gene_sets != ""]
setorder(hub_voc_related, organ, module, -abs_kME_own_module)

fwrite(hub_voc_related, file.path(out_dir, "key_module_voc_related_hub_genes.tsv"), sep = "\t")

hub_voc_top30 <- hub_voc_related[
  ,
  head(.SD, 30),
  by = .(organ, module, module_id)
]

fwrite(hub_voc_top30, file.path(out_dir, "key_module_top30_voc_related_hub_genes.tsv"), sep = "\t")

# ------------------------------------------------------------------
# 5. Module-level summary: size, enrichment, top VOC-related hubs
# ------------------------------------------------------------------

module_size <- all_mem[module_id %in% key_module_summary$module_id, .N, by = .(organ, module, module_id)]
voc_gene_count <- hub_voc_related[, .N, by = .(organ, module, module_id)]
setnames(voc_gene_count, "N", "n_voc_related_genes")

top_enrich <- enrich[
  ,
  .(
    top_enriched_gene_sets = paste(head(gene_set[padj_BH < 0.05 & overlap_genes >= 3], 5), collapse = ";"),
    n_significant_enriched_gene_sets = sum(padj_BH < 0.05 & overlap_genes >= 3, na.rm = TRUE)
  ),
  by = .(organ, module, module_id)
]

top_hubs <- hub_voc_related[
  ,
  .(
    top_voc_related_hub_genes = paste(head(Geneid, 10), collapse = ";"),
    top_voc_related_hub_kME = paste(round(head(abs_kME_own_module, 10), 3), collapse = ";")
  ),
  by = .(organ, module, module_id)
]

module_summary <- merge(key_module_summary, module_size, by = c("organ", "module", "module_id"), all.x = TRUE)
module_summary <- merge(module_summary, voc_gene_count, by = c("organ", "module", "module_id"), all.x = TRUE)
module_summary <- merge(module_summary, top_enrich, by = c("organ", "module", "module_id"), all.x = TRUE)
module_summary <- merge(module_summary, top_hubs, by = c("organ", "module", "module_id"), all.x = TRUE)

module_summary[is.na(n_voc_related_genes), n_voc_related_genes := 0]
module_summary[is.na(top_enriched_gene_sets), top_enriched_gene_sets := ""]
module_summary[is.na(n_significant_enriched_gene_sets), n_significant_enriched_gene_sets := 0]
module_summary[is.na(top_voc_related_hub_genes), top_voc_related_hub_genes := ""]

setorder(module_summary, organ, module)

fwrite(module_summary, file.path(out_dir, "key_module_summary_with_enrichment_and_hubs.tsv"), sep = "\t")

# ------------------------------------------------------------------
# 6. Optional: annotate candidate chains with enriched gene sets and top hubs
# ------------------------------------------------------------------

if (file.exists(candidate_top_file)) {
  cand <- fread(candidate_top_file)
  if (nrow(cand) > 0) {
    cand[, leaf_module_color := clean_module_name(leaf_module)]
    cand[, peel_module_color := clean_module_name(peel_module)]
    cand[, leaf_module_id := module_id("Leaf", leaf_module_color)]
    cand[, peel_module_id := module_id("Peel", peel_module_color)]

    leaf_summary_map <- module_summary[organ == "Leaf",
                                       .(leaf_module_id = module_id,
                                         leaf_top_enriched_gene_sets = top_enriched_gene_sets,
                                         leaf_top_voc_related_hub_genes = top_voc_related_hub_genes)]

    peel_summary_map <- module_summary[organ == "Peel",
                                       .(peel_module_id = module_id,
                                         peel_top_enriched_gene_sets = top_enriched_gene_sets,
                                         peel_top_voc_related_hub_genes = top_voc_related_hub_genes)]

    cand2 <- merge(cand, leaf_summary_map, by = "leaf_module_id", all.x = TRUE)
    cand2 <- merge(cand2, peel_summary_map, by = "peel_module_id", all.x = TRUE)

    fwrite(cand2, file.path(out_dir, "candidate_module_voc_chains_annotated_with_hubs.tsv"), sep = "\t")
  }
}

# ------------------------------------------------------------------
# 7. Summary
# ------------------------------------------------------------------

summary <- data.table(
  item = c(
    "n_selected_key_modules",
    "n_selected_leaf_modules",
    "n_selected_peel_modules",
    "n_enrichment_tests",
    "n_significant_enrichments_fdr005_overlap3",
    "n_key_module_genes_total",
    "n_key_module_voc_related_genes_total"
  ),
  value = c(
    nrow(key_module_summary),
    nrow(key_module_summary[organ == "Leaf"]),
    nrow(key_module_summary[organ == "Peel"]),
    nrow(enrich),
    nrow(enrich_sig),
    nrow(hub_all),
    nrow(hub_voc_related)
  )
)

fwrite(summary, file.path(out_dir, "module_gene_enrichment_summary.tsv"), sep = "\t")

message("\nSummary:")
print(summary)
message("\nDone.")
