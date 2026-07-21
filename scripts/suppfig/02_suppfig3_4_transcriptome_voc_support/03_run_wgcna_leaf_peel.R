suppressPackageStartupMessages({
  library(WGCNA)
  library(data.table)
  library(ggplot2)
  library(pheatmap)
})

options(stringsAsFactors = FALSE)

project <- Sys.getenv("PROJECT_DIR", unset = ".")
project <- normalizePath(project, mustWork = FALSE)
input_dir <- file.path(project, "analysis/wgcna_inputs")
out_base <- file.path(project, "analysis/wgcna_modules")
fig_base <- file.path(project, "figures/wgcna_modules")

dir.create(out_base, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_base, recursive = TRUE, showWarnings = FALSE)

# Use multiple threads if supported
try(enableWGCNAThreads(nThreads = 24), silent = TRUE)

select_power <- function(fit_df) {
  fit_df <- as.data.frame(fit_df)
  fit_df <- fit_df[is.finite(fit_df$SFT.R.sq), ]

  cand80 <- fit_df$Power[fit_df$SFT.R.sq >= 0.80]
  if (length(cand80) > 0) return(min(cand80))

  cand70 <- fit_df$Power[fit_df$SFT.R.sq >= 0.70]
  if (length(cand70) > 0) return(min(cand70))

  fit_df$Power[which.max(fit_df$SFT.R.sq)]
}

run_one_organ <- function(organ) {
  organ_lower <- tolower(organ)

  message("\n============================================================")
  message("Running WGCNA for: ", organ)
  message("============================================================")

  out_dir <- file.path(out_base, organ_lower)
  fig_dir <- file.path(fig_base, organ_lower)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

  expr_file <- file.path(input_dir, paste0(organ_lower, "_vst_top12000_sample_expression.tsv"))

  dat0 <- fread(expr_file, data.table = FALSE)
  message("Input file: ", expr_file)
  message("Input shape: ", paste(dim(dat0), collapse = " x "))

  meta_cols <- c("sample_id", "group_id", "cultivar", "stage", "replicate", "organ")
  missing_meta <- setdiff(meta_cols, colnames(dat0))
  if (length(missing_meta) > 0) {
    stop("Missing metadata columns: ", paste(missing_meta, collapse = ", "))
  }

  meta <- dat0[, meta_cols]
  gene_cols <- setdiff(colnames(dat0), meta_cols)

  datExpr <- dat0[, gene_cols]
  datExpr <- as.data.frame(lapply(datExpr, as.numeric))
  rownames(datExpr) <- meta$sample_id

  message("Expression matrix before filtering: ", paste(dim(datExpr), collapse = " x "))

  # Remove zero-variance or non-finite genes
  gene_var <- apply(datExpr, 2, var, na.rm = TRUE)
  keep_genes <- is.finite(gene_var) & gene_var > 0
  datExpr <- datExpr[, keep_genes, drop = FALSE]

  message("Expression matrix after zero-variance filtering: ", paste(dim(datExpr), collapse = " x "))

  gsg <- goodSamplesGenes(datExpr, verbose = 3)
  if (!gsg$allOK) {
    if (sum(!gsg$goodGenes) > 0) {
      message("Removing genes failing goodSamplesGenes: ", sum(!gsg$goodGenes))
      datExpr <- datExpr[, gsg$goodGenes, drop = FALSE]
    }
    if (sum(!gsg$goodSamples) > 0) {
      message("Removing samples failing goodSamplesGenes: ", sum(!gsg$goodSamples))
      datExpr <- datExpr[gsg$goodSamples, , drop = FALSE]
      meta <- meta[match(rownames(datExpr), meta$sample_id), ]
    }
  }

  message("Expression matrix after goodSamplesGenes: ", paste(dim(datExpr), collapse = " x "))

  # Sample clustering QC
  pdf(file.path(fig_dir, paste0(organ_lower, "_sample_clustering.pdf")), width = 10, height = 6)
  sampleTree <- hclust(dist(datExpr), method = "average")
  plot(sampleTree, main = paste0(organ, " sample clustering"),
       sub = "", xlab = "", cex.lab = 1.2, cex.axis = 1.2, cex.main = 1.2)
  dev.off()

  # Soft-threshold power selection
  powers <- c(1:20, seq(22, 30, by = 2))
  networkType <- "signed"

  message("Picking soft-thresholding power...")
  sft <- pickSoftThreshold(
    datExpr,
    powerVector = powers,
    networkType = networkType,
    corFnc = "bicor",
    corOptions = list(maxPOutliers = 0.05, use = "pairwise.complete.obs"),
    verbose = 5
  )

  fit_df <- as.data.frame(sft$fitIndices)
  softPower <- select_power(fit_df)

  message("Selected soft power: ", softPower)

  fwrite(fit_df, file.path(out_dir, paste0(organ_lower, "_soft_power_fit_indices.tsv")), sep = "\t")

  writeLines(
    c(
      paste0("organ\t", organ),
      paste0("networkType\t", networkType),
      paste0("corType\tbicor"),
      paste0("selected_power\t", softPower)
    ),
    con = file.path(out_dir, paste0(organ_lower, "_selected_soft_power.txt"))
  )

  pdf(file.path(fig_dir, paste0(organ_lower, "_soft_power_diagnostics.pdf")), width = 10, height = 5)
  par(mfrow = c(1, 2))
  plot(fit_df$Power, fit_df$SFT.R.sq,
       xlab = "Soft threshold power",
       ylab = "Scale-free topology fit index",
       type = "b",
       main = paste0(organ, ": scale independence"))
  abline(h = 0.80, col = "red", lty = 2)
  abline(v = softPower, col = "blue", lty = 2)

  plot(fit_df$Power, fit_df$mean.k.,
       xlab = "Soft threshold power",
       ylab = "Mean connectivity",
       type = "b",
       main = paste0(organ, ": mean connectivity"))
  abline(v = softPower, col = "blue", lty = 2)
  dev.off()

  # WGCNA module construction
  message("Running blockwiseModules...")

  net <- blockwiseModules(
    datExpr,
    power = softPower,
    networkType = "signed",
    TOMType = "signed",
    corType = "bicor",
    maxPOutliers = 0.05,
    maxBlockSize = ncol(datExpr),
    minModuleSize = 30,
    reassignThreshold = 0,
    mergeCutHeight = 0.25,
    numericLabels = FALSE,
    pamRespectsDendro = FALSE,
    saveTOMs = FALSE,
    verbose = 3
  )

  moduleColors <- net$colors
  names(moduleColors) <- colnames(datExpr)

  module_size <- as.data.frame(table(moduleColors))
  colnames(module_size) <- c("module", "n_genes")
  module_size <- module_size[order(module_size$n_genes, decreasing = TRUE), ]

  fwrite(module_size, file.path(out_dir, paste0(organ_lower, "_module_size.tsv")), sep = "\t")

  gene_module <- data.frame(
    Geneid = names(moduleColors),
    module = unname(moduleColors),
    stringsAsFactors = FALSE
  )

  # Module eigengenes
  MEs0 <- moduleEigengenes(datExpr, colors = moduleColors)$eigengenes
  MEs <- orderMEs(MEs0)

  me_df <- cbind(
    meta[match(rownames(MEs), meta$sample_id), ],
    as.data.frame(MEs)
  )

  fwrite(me_df, file.path(out_dir, paste0(organ_lower, "_module_eigengenes_sample_level.tsv")), sep = "\t")

  me_cols <- grep("^ME", colnames(me_df), value = TRUE)

  me_dt <- as.data.table(me_df)
  me_profile <- me_dt[, lapply(.SD, mean), by = .(group_id, cultivar, stage, organ), .SDcols = me_cols]
  setorder(me_profile, cultivar, stage)

  fwrite(me_profile, file.path(out_dir, paste0(organ_lower, "_module_eigengenes_profile_52profiles.tsv")), sep = "\t")

  # Module membership / kME
  message("Calculating module membership kME...")
  kME <- signedKME(
    datExpr,
    MEs,
    outputColumnName = "kME",
    corFnc = "bicor",
    corOptions = "maxPOutliers = 0.05, use = 'pairwise.complete.obs'"
  )

  kME <- as.data.frame(kME)
  kME$Geneid <- rownames(kME)

  gene_module <- merge(gene_module, kME, by = "Geneid", all.x = TRUE)

  own_kme <- rep(NA_real_, nrow(gene_module))
  for (i in seq_len(nrow(gene_module))) {
    col_i <- paste0("kME", gene_module$module[i])
    if (col_i %in% colnames(gene_module)) {
      own_kme[i] <- gene_module[[col_i]][i]
    }
  }

  gene_module$kME_own_module <- own_kme

  gene_module <- gene_module[, c("Geneid", "module", "kME_own_module",
                                 setdiff(colnames(gene_module), c("Geneid", "module", "kME_own_module")))]

  fwrite(gene_module, file.path(out_dir, paste0(organ_lower, "_gene_module_membership.tsv")), sep = "\t")

  # Dendrogram and module colors
  pdf(file.path(fig_dir, paste0(organ_lower, "_module_dendrogram.pdf")), width = 12, height = 7)
  for (b in seq_along(net$dendrograms)) {
    block_genes <- net$blockGenes[[b]]
    plotDendroAndColors(
      net$dendrograms[[b]],
      moduleColors[block_genes],
      "Module colors",
      dendroLabels = FALSE,
      hang = 0.03,
      addGuide = TRUE,
      guideHang = 0.05,
      main = paste0(organ, " WGCNA modules, block ", b)
    )
  }
  dev.off()

  save(
    net, datExpr, meta, moduleColors, MEs, softPower, fit_df, module_size,
    file = file.path(out_dir, paste0(organ_lower, "_wgcna_result.RData"))
  )

  message("\nCompleted WGCNA for ", organ)
  message("Module count including grey: ", nrow(module_size))
  message("Non-grey module count: ", sum(module_size$module != "grey"))
  message("Largest modules:")
  print(head(module_size, 10))

  return(list(
    organ = organ,
    n_samples = nrow(datExpr),
    n_genes = ncol(datExpr),
    selected_power = softPower,
    n_modules_total = nrow(module_size),
    n_modules_non_grey = sum(module_size$module != "grey")
  ))
}

leaf_summary <- run_one_organ("Leaf")
peel_summary <- run_one_organ("Peel")

summary <- rbind(
  as.data.frame(leaf_summary),
  as.data.frame(peel_summary)
)

fwrite(summary, file.path(out_base, "wgcna_run_summary.tsv"), sep = "\t")

message("\n============================================================")
message("WGCNA finished for Leaf and Peel")
message("============================================================")
print(summary)
