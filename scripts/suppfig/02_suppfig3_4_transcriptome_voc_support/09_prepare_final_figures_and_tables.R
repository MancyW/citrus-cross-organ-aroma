suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(pheatmap)
})

options(stringsAsFactors = FALSE)

project <- Sys.getenv("PROJECT_DIR", unset = ".")
project <- normalizePath(project, mustWork = FALSE)

pkg_dir <- file.path(project, "analysis/final_evidence_package")
out_dir <- file.path(project, "analysis/final_figure_package")
fig_dir <- file.path(project, "figures/final_figure_package")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

read_file <- function(name) {
  path <- file.path(pkg_dir, name)
  if (!file.exists(path)) {
    warning("Missing file: ", path)
    return(data.table())
  }
  fread(path)
}

short_module <- function(x) {
  x <- gsub("^LeafME__", "L_", x)
  x <- gsub("^PeelME__", "P_", x)
  x
}

short_trait <- function(x) {
  x <- gsub("^PeelVOC__", "P_", x)
  x <- gsub("^LeafVOC__", "L_", x)
  x <- gsub("_", "-", x)
  x
}

# ------------------------------------------------------------
# 1. Read final evidence package tables
# ------------------------------------------------------------

module_pairs <- read_file("top_leaf_peel_module_pairs.tsv")
module_pairs_resid <- read_file("top_leaf_peel_module_pairs_stage_residual.tsv")
module_voc <- read_file("top_module_peel_voc_associations.tsv")
module_voc_resid <- read_file("top_module_peel_voc_associations_stage_residual.tsv")
chains <- read_file("top_candidate_module_voc_chains.tsv")
triplets <- read_file("candidate_triplets_manuscript_examples.tsv")
metrics <- read_file("final_evidence_key_metrics.tsv")
key_modules <- read_file("key_module_summary_with_hubs.tsv")

# ------------------------------------------------------------
# 2. Select representative VOCs and chains for manuscript/figure
# ------------------------------------------------------------

priority_vocs <- c(
  "Decanal",
  "(E)-β-Ocimene",
  "Linalool",
  "Citronellol",
  "β-Myrcene",
  "D-Limonene",
  "Undecanal",
  "Dodecanal"
)

if (nrow(chains) > 0) {
  chains[, voc_priority := match(peel_voc_original, priority_vocs)]
  chains[is.na(voc_priority), voc_priority := 999]
  chains[, evidence_score := as.numeric(evidence_score)]

  chains_sel <- chains[
    order(voc_priority, -evidence_score)
  ][
    ,
    head(.SD, 4),
    by = peel_voc_original
  ]

  chains_sel <- chains_sel[order(voc_priority, -evidence_score)]
  fwrite(chains_sel, file.path(out_dir, "selected_candidate_module_voc_chains_for_figure.tsv"), sep = "\t")
} else {
  chains_sel <- data.table()
}

if (nrow(triplets) > 0) {
  triplets[, voc_priority := match(peel_voc_original, priority_vocs)]
  triplets[is.na(voc_priority), voc_priority := 999]
  triplets[, evidence_score := as.numeric(evidence_score)]

  triplets_sel <- triplets[
    order(voc_priority, evidence_tier, -evidence_score)
  ][
    ,
    head(.SD, 5),
    by = peel_voc_original
  ]

  triplets_sel <- triplets_sel[order(voc_priority, evidence_tier, -evidence_score)]
  fwrite(triplets_sel, file.path(out_dir, "selected_candidate_gene_triplets_for_figure.tsv"), sep = "\t")
} else {
  triplets_sel <- data.table()
}

# ------------------------------------------------------------
# 3. Selected leaf-peel module pair heatmap
# ------------------------------------------------------------

if (nrow(chains_sel) > 0 && nrow(module_pairs) > 0) {
  leaf_mods <- unique(chains_sel$leaf_module)
  peel_mods <- unique(chains_sel$peel_module)

  mm_plot <- module_pairs[left %in% leaf_mods & right %in% peel_mods]
  mat_dt <- dcast(mm_plot, left ~ right, value.var = "cor")
  mat <- as.matrix(mat_dt[, -1, drop = FALSE])
  rownames(mat) <- short_module(mat_dt$left)
  colnames(mat) <- short_module(colnames(mat))

  mat[!is.finite(mat)] <- 0

  pdf(file.path(fig_dir, "FigY_selected_leaf_peel_module_correlation_heatmap.pdf"),
      width = max(6, ncol(mat) * 0.55),
      height = max(5, nrow(mat) * 0.45))
  pheatmap(
    mat,
    main = "Selected leaf–peel module correlations",
    cluster_rows = TRUE,
    cluster_cols = TRUE,
    border_color = NA,
    fontsize_row = 8,
    fontsize_col = 8
  )
  dev.off()
}

# ------------------------------------------------------------
# 4. Selected module-peel VOC heatmap
# ------------------------------------------------------------

if (nrow(chains_sel) > 0 && nrow(module_voc) > 0) {
  sel_modules <- unique(c(chains_sel$leaf_module, chains_sel$peel_module))
  sel_vocs <- unique(chains_sel$peel_voc_original)

  mv_plot <- module_voc[
    left %in% sel_modules &
      trait_original %in% sel_vocs
  ]

  mat_dt <- dcast(mv_plot, left ~ trait_original, value.var = "cor")
  mat <- as.matrix(mat_dt[, -1, drop = FALSE])
  rownames(mat) <- short_module(mat_dt$left)
  colnames(mat) <- colnames(mat)

  mat[!is.finite(mat)] <- 0

  pdf(file.path(fig_dir, "FigY_selected_modules_vs_peel_voc_heatmap.pdf"),
      width = max(6, ncol(mat) * 0.75),
      height = max(5, nrow(mat) * 0.42))
  pheatmap(
    mat,
    main = "Selected modules vs peel VOC traits",
    cluster_rows = TRUE,
    cluster_cols = TRUE,
    border_color = NA,
    fontsize_row = 8,
    fontsize_col = 8
  )
  dev.off()
}

# ------------------------------------------------------------
# 5. Candidate module–VOC chain network
# ------------------------------------------------------------

if (nrow(chains_sel) > 0) {
  net_chains <- copy(chains_sel)

  # Keep a manageable number for visual clarity
  net_chains <- net_chains[order(voc_priority, -evidence_score)]
  net_chains <- head(net_chains, 24)

  leaf_nodes <- data.table(
    id = unique(net_chains$leaf_module),
    layer = "Leaf module",
    x = 1
  )
  peel_nodes <- data.table(
    id = unique(net_chains$peel_module),
    layer = "Peel module",
    x = 2
  )
  voc_nodes <- data.table(
    id = unique(net_chains$peel_voc_original),
    layer = "Peel VOC",
    x = 3
  )

  nodes <- rbindlist(list(leaf_nodes, peel_nodes, voc_nodes), fill = TRUE)

  nodes[, y := seq(.N, 1), by = layer]
  nodes[, label := fifelse(layer == "Peel VOC", id, short_module(id))]

  # Normalize y within layer to similar visual span
  nodes[, y := if (.N == 1) 0.5 else (y - min(y)) / (max(y) - min(y)), by = layer]

  e1 <- net_chains[, .(
    from = leaf_module,
    to = peel_module,
    edge_type = "module-module",
    cor = as.numeric(leaf_peel_module_cor),
    evidence_score = as.numeric(evidence_score),
    voc = peel_voc_original
  )]

  e2 <- net_chains[, .(
    from = peel_module,
    to = peel_voc_original,
    edge_type = "module-VOC",
    cor = as.numeric(peel_module_peel_voc_cor),
    evidence_score = as.numeric(evidence_score),
    voc = peel_voc_original
  )]

  edges <- rbindlist(list(e1, e2), fill = TRUE)
  edges <- merge(edges, nodes[, .(from = id, x_from = x, y_from = y)], by = "from", all.x = TRUE)
  edges <- merge(edges, nodes[, .(to = id, x_to = x, y_to = y)], by = "to", all.x = TRUE)

  edges[, sign_label := ifelse(cor >= 0, "positive", "negative")]
  edges[, line_width := 0.2 + 1.5 * abs(cor)]

  p_net <- ggplot() +
    geom_segment(
      data = edges,
      aes(x = x_from, y = y_from, xend = x_to, yend = y_to,
          linewidth = line_width, linetype = sign_label),
      alpha = 0.65
    ) +
    geom_point(
      data = nodes,
      aes(x = x, y = y),
      size = 3
    ) +
    geom_text(
      data = nodes,
      aes(x = x, y = y, label = label),
      hjust = ifelse(nodes$x == 3, 0, 0.5),
      vjust = -0.8,
      size = 3
    ) +
    scale_x_continuous(
      breaks = c(1, 2, 3),
      labels = c("Leaf module", "Peel module", "Peel VOC"),
      limits = c(0.7, 3.45)
    ) +
    guides(linewidth = "none") +
    labs(
      title = "Candidate leaf module–peel module–peel VOC chains",
      x = NULL,
      y = NULL,
      linetype = "Correlation sign"
    ) +
    theme_bw() +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid = element_blank()
    )

  ggsave(
    file.path(fig_dir, "FigY_candidate_module_voc_chain_network.pdf"),
    p_net,
    width = 9,
    height = 6
  )

  fwrite(net_chains, file.path(out_dir, "candidate_module_voc_chains_used_for_network.tsv"), sep = "\t")
}

# ------------------------------------------------------------
# 6. Candidate gene triplet lollipop plot
# ------------------------------------------------------------

if (nrow(triplets_sel) > 0) {
  plot_dt <- copy(triplets_sel)
  plot_dt <- plot_dt[order(voc_priority, evidence_tier, -evidence_score)]
  plot_dt <- head(plot_dt, 30)

  plot_dt[, label := paste0(
    leaf_gene, " → ", peel_gene, " / ", peel_voc_original
  )]

  plot_dt[, label := factor(label, levels = rev(unique(label)))]

  p_trip <- ggplot(plot_dt, aes(x = as.numeric(evidence_score), y = label)) +
    geom_segment(aes(x = 0, xend = as.numeric(evidence_score), y = label, yend = label)) +
    geom_point(size = 2.2) +
    labs(
      title = "Representative candidate leaf gene–peel gene–peel VOC triplets",
      x = "Evidence score",
      y = NULL
    ) +
    theme_bw() +
    theme(
      axis.text.y = element_text(size = 6),
      panel.grid.minor = element_blank()
    )

  ggsave(
    file.path(fig_dir, "FigY_candidate_gene_triplet_evidence_score_lollipop.pdf"),
    p_trip,
    width = 10,
    height = 8
  )

  fwrite(plot_dt, file.path(out_dir, "candidate_gene_triplets_used_for_lollipop.tsv"), sep = "\t")
}

# ------------------------------------------------------------
# 7. Figure source inventory and final notes
# ------------------------------------------------------------

source_inventory <- data.table(
  proposed_panel = c(
    "FigX-A",
    "FigX-B",
    "FigX-C",
    "FigX-D",
    "FigX-E",
    "FigY-A",
    "FigY-B",
    "FigY-C",
    "FigY-D",
    "FigY-E",
    "FigY-F"
  ),
  content = c(
    "RNA-seq PCA of all 416 samples",
    "PERMANOVA R2 contributions",
    "Global leaf-peel transcriptome distance correspondence",
    "Stage-specific global Mantel r",
    "VOC/metabolism gene-set coordination with random background",
    "Leaf WGCNA module dendrogram",
    "Peel WGCNA module dendrogram",
    "Selected leaf module-peel module correlation heatmap",
    "Selected module-peel VOC heatmap",
    "Candidate module-VOC chain network",
    "Representative candidate gene triplets"
  ),
  current_source = c(
    "figures/rnaseq_qc/",
    "analysis/rnaseq_qc/PERMANOVA_by_terms_all_samples.txt + redraw needed",
    "figures/leaf_peel_coordination/leaf_peel_transcriptome_distance_scatter.pdf",
    "figures/transcriptome_support_summary/global_stage_mantel_barplot.pdf",
    "figures/transcriptome_support_summary/pathway_mantel_with_random_background.pdf",
    "figures/wgcna_modules/leaf/leaf_module_dendrogram.pdf",
    "figures/wgcna_modules/peel/peel_module_dendrogram.pdf",
    "figures/final_figure_package/FigY_selected_leaf_peel_module_correlation_heatmap.pdf",
    "figures/final_figure_package/FigY_selected_modules_vs_peel_voc_heatmap.pdf",
    "figures/final_figure_package/FigY_candidate_module_voc_chain_network.pdf",
    "figures/final_figure_package/FigY_candidate_gene_triplet_evidence_score_lollipop.pdf"
  ),
  status = c(
    "existing, may need final styling",
    "needs redraw",
    "existing",
    "existing",
    "existing",
    "existing",
    "existing",
    "generated in Step 24",
    "generated in Step 24",
    "generated in Step 24",
    "generated in Step 24"
  )
)

fwrite(source_inventory, file.path(out_dir, "supplementary_figure_source_inventory.tsv"), sep = "\t")

notes <- c(
  "# Final figure package notes",
  "",
  "## Recommended final figure structure",
  "",
  "### Supplementary Figure X",
  "Cross-organ transcriptomic coordination between leaf and peel.",
  "",
  "Recommended panels:",
  "A. RNA-seq PCA.",
  "B. PERMANOVA R2 contributions.",
  "C. Global leaf-peel transcriptome distance correspondence.",
  "D. Stage-specific global Mantel r.",
  "E. VOC/metabolism gene-set coordination with random gene-set background.",
  "",
  "### Supplementary Figure Y",
  "Module-level and candidate gene-level support linking leaf and peel transcriptomes to peel VOC traits.",
  "",
  "Recommended panels:",
  "A. Leaf WGCNA module dendrogram.",
  "B. Peel WGCNA module dendrogram.",
  "C. Selected leaf-peel module correlation heatmap.",
  "D. Selected module-peel VOC heatmap.",
  "E. Candidate module-VOC chain network.",
  "F. Representative candidate gene triplets.",
  "",
  "## Interpretation boundary",
  "",
  "Use these results as association-level molecular support. Avoid causal language such as leaf genes regulating or determining peel VOC biosynthesis.",
  "",
  "Preferred wording:",
  "",
  "Cross-organ co-expression analysis identified coordinated leaf and peel modules associated with key peel VOC traits. Representative module-VOC chains were further supported by candidate leaf hub gene-peel hub gene-peel VOC triplets, providing association-level molecular support for the leaf-peel volatile proxy framework."
)

writeLines(notes, file.path(out_dir, "final_figure_package_notes.md"))

message("Generated final figure package:")
message(out_dir)
message(fig_dir)
