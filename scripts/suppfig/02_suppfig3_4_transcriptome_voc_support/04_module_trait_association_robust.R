suppressPackageStartupMessages({
  library(data.table)
  library(WGCNA)
  library(pheatmap)
})

options(stringsAsFactors = FALSE)

project <- Sys.getenv("PROJECT_DIR", unset = ".")
project <- normalizePath(project, mustWork = FALSE)

wgcna_dir <- file.path(project, "analysis/wgcna_modules")
input_dir <- file.path(project, "analysis/wgcna_inputs")
out_dir <- file.path(project, "analysis/module_trait_association")
fig_dir <- file.path(project, "figures/module_trait_association")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

leaf_me_file <- file.path(wgcna_dir, "leaf/leaf_module_eigengenes_profile_52profiles.tsv")
peel_me_file <- file.path(wgcna_dir, "peel/peel_module_eigengenes_profile_52profiles.tsv")
trait_key_file <- file.path(input_dir, "traits_leaf_peel_key_voc_clip0_log1p_52profiles.safe.tsv")
trait_map_file <- file.path(input_dir, "voc_trait_name_map.safe.tsv")

read_module_eigengenes <- function(path, prefix) {
  df <- fread(path, data.table = FALSE)
  me_cols <- grep("^ME", colnames(df), value = TRUE)
  me_cols <- setdiff(me_cols, "MEgrey")

  out <- df[, c("group_id", "cultivar", "stage", me_cols), drop = FALSE]
  new_me_cols <- paste0(prefix, "__", sub("^ME", "", me_cols))
  colnames(out)[match(me_cols, colnames(out))] <- new_me_cols
  out
}

corr_table <- function(X, Y, x_cols, y_cols, method = "spearman", min_n = 10) {
  rows <- vector("list", length(x_cols) * length(y_cols))
  k <- 1

  for (xc in x_cols) {
    x <- as.numeric(X[[xc]])

    for (yc in y_cols) {
      y <- as.numeric(Y[[yc]])
      ok <- is.finite(x) & is.finite(y)
      n_ok <- sum(ok)

      if (n_ok >= min_n && sd(x[ok]) > 0 && sd(y[ok]) > 0) {
        ct <- suppressWarnings(cor.test(x[ok], y[ok], method = method, exact = FALSE))
        r <- unname(ct$estimate)
        p <- ct$p.value
      } else {
        r <- NA_real_
        p <- NA_real_
      }

      rows[[k]] <- data.frame(
        left = xc,
        right = yc,
        n = n_ok,
        method = method,
        cor = r,
        p_value = p,
        stringsAsFactors = FALSE
      )
      k <- k + 1
    }
  }

  out <- rbindlist(rows)
  out$padj_BH <- p.adjust(out$p_value, method = "BH")
  out <- out[order(out$padj_BH, -abs(out$cor)), ]
  as.data.frame(out)
}

residualize_by_stage <- function(df, value_cols, stage_vector) {
  out <- as.data.frame(df[, value_cols, drop = FALSE])
  stage_factor <- factor(stage_vector, levels = c("S1", "S2", "S3", "S4"))

  for (cc in value_cols) {
    v <- as.numeric(out[[cc]])
    if (sd(v, na.rm = TRUE) > 0) {
      out[[cc]] <- residuals(lm(v ~ stage_factor))
    } else {
      out[[cc]] <- NA_real_
    }
  }
  out
}

make_heatmap_robust <- function(df, outfile, title) {
  if (nrow(df) == 0) {
    message("Skip heatmap, empty input: ", outfile)
    return(invisible(NULL))
  }

  mat_dt <- dcast(as.data.table(df), left ~ right, value.var = "cor")
  mat <- as.matrix(mat_dt[, -1, drop = FALSE])
  rownames(mat) <- mat_dt$left

  # Remove all-NA rows/columns
  keep_rows <- rowSums(is.finite(mat)) > 0
  keep_cols <- colSums(is.finite(mat)) > 0
  mat <- mat[keep_rows, keep_cols, drop = FALSE]

  if (nrow(mat) < 2 || ncol(mat) < 2) {
    message("Skip heatmap, too few finite rows/cols: ", outfile)
    return(invisible(NULL))
  }

  # Fill remaining NA for plotting only
  mat[!is.finite(mat)] <- 0

  rownames(mat) <- gsub("^LeafME__", "L_", rownames(mat))
  rownames(mat) <- gsub("^PeelME__", "P_", rownames(mat))
  colnames(mat) <- gsub("^LeafME__", "L_", colnames(mat))
  colnames(mat) <- gsub("^PeelME__", "P_", colnames(mat))
  colnames(mat) <- gsub("^PeelVOC__", "P_", colnames(mat))
  colnames(mat) <- gsub("^LeafVOC__", "L_", colnames(mat))

  pdf(outfile, width = max(7, ncol(mat) * 0.42), height = max(6, nrow(mat) * 0.28))
  pheatmap(
    mat,
    main = title,
    cluster_rows = TRUE,
    cluster_cols = TRUE,
    fontsize_row = 7,
    fontsize_col = 7,
    border_color = NA
  )
  dev.off()

  message("Wrote heatmap: ", outfile)
}

write_and_report <- function(df, file, top_n = 20) {
  fwrite(df, file, sep = "\t")
  message("Wrote: ", file, "  rows=", nrow(df))
  print(head(df, top_n))
}

leaf_me <- read_module_eigengenes(leaf_me_file, "LeafME")
peel_me <- read_module_eigengenes(peel_me_file, "PeelME")
traits <- fread(trait_key_file, data.table = FALSE)

common_profiles <- Reduce(intersect, list(leaf_me$group_id, peel_me$group_id, traits$group_id))
common_profiles <- traits$group_id[traits$group_id %in% common_profiles]

leaf_me <- leaf_me[match(common_profiles, leaf_me$group_id), ]
peel_me <- peel_me[match(common_profiles, peel_me$group_id), ]
traits <- traits[match(common_profiles, traits$group_id), ]

if (!all(leaf_me$group_id == peel_me$group_id) || !all(leaf_me$group_id == traits$group_id)) {
  stop("Profile order mismatch.")
}

leaf_module_cols <- grep("^LeafME__", colnames(leaf_me), value = TRUE)
peel_module_cols <- grep("^PeelME__", colnames(peel_me), value = TRUE)
leaf_voc_cols <- grep("^LeafVOC__", colnames(traits), value = TRUE)
peel_voc_cols <- grep("^PeelVOC__", colnames(traits), value = TRUE)
all_voc_cols <- c(leaf_voc_cols, peel_voc_cols)

message("Profiles: ", length(common_profiles))
message("Leaf modules: ", length(leaf_module_cols))
message("Peel modules: ", length(peel_module_cols))
message("Peel key VOC traits: ", length(peel_voc_cols))

all_module_df <- cbind(
  leaf_me[, c("group_id", "cultivar", "stage", leaf_module_cols), drop = FALSE],
  peel_me[, peel_module_cols, drop = FALSE]
)
all_module_cols <- c(leaf_module_cols, peel_module_cols)

# ------------------------------------------------------------------
# 1. Recompute core overall associations
# ------------------------------------------------------------------

module_module_overall <- corr_table(
  leaf_me,
  peel_me,
  leaf_module_cols,
  peel_module_cols,
  method = "spearman",
  min_n = 30
)

write_and_report(
  module_module_overall,
  file.path(out_dir, "leaf_module_vs_peel_module_spearman_overall.tsv")
)

make_heatmap_robust(
  module_module_overall,
  file.path(fig_dir, "leaf_module_vs_peel_module_spearman_overall_heatmap.robust.pdf"),
  "Leaf module vs peel module correlation"
)

module_voc_overall <- corr_table(
  all_module_df,
  traits,
  all_module_cols,
  all_voc_cols,
  method = "spearman",
  min_n = 30
)

module_voc_overall$module_organ <- ifelse(grepl("^LeafME__", module_voc_overall$left), "Leaf", "Peel")
module_voc_overall$trait_organ <- ifelse(grepl("^LeafVOC__", module_voc_overall$right), "Leaf", "Peel")

write_and_report(
  module_voc_overall,
  file.path(out_dir, "module_vs_key_voc_spearman_overall.tsv")
)

module_voc_peel_only <- module_voc_overall[grepl("^PeelVOC__", module_voc_overall$right), ]

write_and_report(
  module_voc_peel_only,
  file.path(out_dir, "module_vs_peel_key_voc_spearman_overall.tsv")
)

make_heatmap_robust(
  module_voc_peel_only[grepl("^LeafME__", module_voc_peel_only$left), ],
  file.path(fig_dir, "leaf_modules_vs_peel_key_voc_spearman_heatmap.robust.pdf"),
  "Leaf modules vs peel key VOC traits"
)

make_heatmap_robust(
  module_voc_peel_only[grepl("^PeelME__", module_voc_peel_only$left), ],
  file.path(fig_dir, "peel_modules_vs_peel_key_voc_spearman_heatmap.robust.pdf"),
  "Peel modules vs peel key VOC traits"
)

# ------------------------------------------------------------------
# 2. Stage-residual associations
# ------------------------------------------------------------------

stage_vector <- traits$stage

leaf_me_resid <- residualize_by_stage(leaf_me, leaf_module_cols, stage_vector)
peel_me_resid <- residualize_by_stage(peel_me, peel_module_cols, stage_vector)

module_module_stage_resid <- corr_table(
  leaf_me_resid,
  peel_me_resid,
  leaf_module_cols,
  peel_module_cols,
  method = "pearson",
  min_n = 30
)

write_and_report(
  module_module_stage_resid,
  file.path(out_dir, "leaf_module_vs_peel_module_stage_residual_pearson.tsv")
)

make_heatmap_robust(
  module_module_stage_resid,
  file.path(fig_dir, "leaf_module_vs_peel_module_stage_residual_pearson_heatmap.robust.pdf"),
  "Leaf module vs peel module correlation after stage residualization"
)

module_resid <- residualize_by_stage(all_module_df, all_module_cols, stage_vector)
trait_resid <- residualize_by_stage(traits, all_voc_cols, stage_vector)

module_voc_stage_resid <- corr_table(
  module_resid,
  trait_resid,
  all_module_cols,
  all_voc_cols,
  method = "pearson",
  min_n = 30
)

module_voc_stage_resid$module_organ <- ifelse(grepl("^LeafME__", module_voc_stage_resid$left), "Leaf", "Peel")
module_voc_stage_resid$trait_organ <- ifelse(grepl("^LeafVOC__", module_voc_stage_resid$right), "Leaf", "Peel")

write_and_report(
  module_voc_stage_resid,
  file.path(out_dir, "module_vs_key_voc_stage_residual_pearson.tsv")
)

module_voc_peel_stage_resid <- module_voc_stage_resid[grepl("^PeelVOC__", module_voc_stage_resid$right), ]

write_and_report(
  module_voc_peel_stage_resid,
  file.path(out_dir, "module_vs_peel_key_voc_stage_residual_pearson.tsv")
)

make_heatmap_robust(
  module_voc_peel_stage_resid[grepl("^LeafME__", module_voc_peel_stage_resid$left), ],
  file.path(fig_dir, "leaf_modules_vs_peel_key_voc_stage_residual_heatmap.robust.pdf"),
  "Leaf modules vs peel VOC traits after stage residualization"
)

make_heatmap_robust(
  module_voc_peel_stage_resid[grepl("^PeelME__", module_voc_peel_stage_resid$left), ],
  file.path(fig_dir, "peel_modules_vs_peel_key_voc_stage_residual_heatmap.robust.pdf"),
  "Peel modules vs peel VOC traits after stage residualization"
)

# ------------------------------------------------------------------
# 3. By-stage associations
# ------------------------------------------------------------------

stage_rows_mm <- list()
stage_rows_mv <- list()

for (st in c("S1", "S2", "S3", "S4")) {
  idx <- which(traits$stage == st)

  tmp_mm <- corr_table(
    leaf_me[idx, , drop = FALSE],
    peel_me[idx, , drop = FALSE],
    leaf_module_cols,
    peel_module_cols,
    method = "spearman",
    min_n = 10
  )
  tmp_mm$stage <- st
  stage_rows_mm[[st]] <- tmp_mm

  tmp_mv <- corr_table(
    all_module_df[idx, , drop = FALSE],
    traits[idx, , drop = FALSE],
    all_module_cols,
    all_voc_cols,
    method = "spearman",
    min_n = 10
  )
  tmp_mv$stage <- st
  tmp_mv$module_organ <- ifelse(grepl("^LeafME__", tmp_mv$left), "Leaf", "Peel")
  tmp_mv$trait_organ <- ifelse(grepl("^LeafVOC__", tmp_mv$right), "Leaf", "Peel")
  stage_rows_mv[[st]] <- tmp_mv
}

module_module_by_stage <- rbindlist(stage_rows_mm)
module_module_by_stage$padj_BH_by_stage_all <- p.adjust(module_module_by_stage$p_value, method = "BH")

write_and_report(
  module_module_by_stage,
  file.path(out_dir, "leaf_module_vs_peel_module_spearman_by_stage.tsv")
)

module_voc_by_stage <- rbindlist(stage_rows_mv)
module_voc_by_stage$padj_BH_by_stage_all <- p.adjust(module_voc_by_stage$p_value, method = "BH")

write_and_report(
  module_voc_by_stage,
  file.path(out_dir, "module_vs_key_voc_spearman_by_stage.tsv")
)

module_voc_peel_by_stage <- module_voc_by_stage[grepl("^PeelVOC__", module_voc_by_stage$right), ]

write_and_report(
  module_voc_peel_by_stage,
  file.path(out_dir, "module_vs_peel_key_voc_spearman_by_stage.tsv")
)

# ------------------------------------------------------------------
# 4. Candidate evidence chains
# ------------------------------------------------------------------

lp <- as.data.table(module_module_overall)
setnames(lp,
         old = c("left", "right", "cor", "p_value", "padj_BH"),
         new = c("leaf_module", "peel_module", "leaf_peel_module_cor",
                 "leaf_peel_module_p", "leaf_peel_module_fdr"))

lv <- as.data.table(module_voc_peel_only[grepl("^LeafME__", module_voc_peel_only$left), ])
setnames(lv,
         old = c("left", "right", "cor", "p_value", "padj_BH"),
         new = c("leaf_module", "peel_voc_trait", "leaf_module_peel_voc_cor",
                 "leaf_module_peel_voc_p", "leaf_module_peel_voc_fdr"))

pv <- as.data.table(module_voc_peel_only[grepl("^PeelME__", module_voc_peel_only$left), ])
setnames(pv,
         old = c("left", "right", "cor", "p_value", "padj_BH"),
         new = c("peel_module", "peel_voc_trait", "peel_module_peel_voc_cor",
                 "peel_module_peel_voc_p", "peel_module_peel_voc_fdr"))

cand <- merge(
  lp[, .(leaf_module, peel_module, leaf_peel_module_cor,
         leaf_peel_module_p, leaf_peel_module_fdr)],
  lv[, .(leaf_module, peel_voc_trait, leaf_module_peel_voc_cor,
         leaf_module_peel_voc_p, leaf_module_peel_voc_fdr)],
  by = "leaf_module",
  allow.cartesian = TRUE
)

cand <- merge(
  cand,
  pv[, .(peel_module, peel_voc_trait, peel_module_peel_voc_cor,
         peel_module_peel_voc_p, peel_module_peel_voc_fdr)],
  by = c("peel_module", "peel_voc_trait"),
  allow.cartesian = TRUE
)

cand[, direction_coherent := sign(leaf_module_peel_voc_cor) ==
       sign(leaf_peel_module_cor * peel_module_peel_voc_cor)]

cand[, evidence_score := abs(leaf_peel_module_cor) *
       abs(leaf_module_peel_voc_cor) *
       abs(peel_module_peel_voc_cor)]

if (file.exists(trait_map_file)) {
  trait_map <- fread(trait_map_file, data.table = FALSE)
  safe_to_original <- setNames(trait_map$original_trait, trait_map$safe_trait)
  cand[, peel_voc_original := safe_to_original[peel_voc_trait]]
} else {
  cand[, peel_voc_original := peel_voc_trait]
}

setorder(cand,
         -direction_coherent,
         leaf_peel_module_fdr,
         leaf_module_peel_voc_fdr,
         peel_module_peel_voc_fdr,
         -evidence_score)

fwrite(cand, file.path(out_dir, "candidate_leaf_module_peel_module_peel_voc_chains_overall.tsv"), sep = "\t")

cand_top_strict <- cand[
  direction_coherent == TRUE &
    leaf_peel_module_fdr < 0.05 &
    leaf_module_peel_voc_fdr < 0.05 &
    peel_module_peel_voc_fdr < 0.05
]

setorder(cand_top_strict, -evidence_score)

fwrite(cand_top_strict, file.path(out_dir, "candidate_leaf_peel_module_voc_chains_top_strict.tsv"), sep = "\t")

cand_top_ranked <- cand[
  direction_coherent == TRUE &
    leaf_peel_module_fdr < 0.05 &
    abs(leaf_module_peel_voc_cor) >= 0.45 &
    abs(peel_module_peel_voc_cor) >= 0.45
]

setorder(cand_top_ranked, -evidence_score)

fwrite(cand_top_ranked, file.path(out_dir, "candidate_leaf_peel_module_voc_chains_top.tsv"), sep = "\t")

message("Candidate chains total: ", nrow(cand))
message("Candidate chains strict: ", nrow(cand_top_strict))
message("Candidate chains top ranked: ", nrow(cand_top_ranked))
print(head(cand_top_ranked, 30))

# ------------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------------

summary <- data.frame(
  item = c(
    "n_profiles",
    "n_leaf_modules_non_grey",
    "n_peel_modules_non_grey",
    "n_key_leaf_voc_traits",
    "n_key_peel_voc_traits",
    "n_leaf_peel_module_tests",
    "n_leaf_peel_module_sig_fdr005_abs05",
    "n_module_peel_voc_tests",
    "n_module_peel_voc_sig_fdr005_abs05",
    "n_candidate_chains_total",
    "n_candidate_chains_strict",
    "n_candidate_chains_top_ranked"
  ),
  value = c(
    length(common_profiles),
    length(leaf_module_cols),
    length(peel_module_cols),
    length(leaf_voc_cols),
    length(peel_voc_cols),
    nrow(module_module_overall),
    sum(module_module_overall$padj_BH < 0.05 & abs(module_module_overall$cor) >= 0.5, na.rm = TRUE),
    nrow(module_voc_peel_only),
    sum(module_voc_peel_only$padj_BH < 0.05 & abs(module_voc_peel_only$cor) >= 0.5, na.rm = TRUE),
    nrow(cand),
    nrow(cand_top_strict),
    nrow(cand_top_ranked)
  )
)

fwrite(summary, file.path(out_dir, "module_trait_association_summary.tsv"), sep = "\t")

message("\nSummary:")
print(summary)
message("\nDone.")
