suppressPackageStartupMessages({
  library(data.table)
})

options(stringsAsFactors = FALSE)

project <- Sys.getenv("PROJECT_DIR", unset = ".")
project <- normalizePath(project, mustWork = FALSE)

input_dir <- file.path(project, "analysis/wgcna_inputs")
enrich_dir <- file.path(project, "analysis/module_gene_enrichment")
out_dir <- file.path(project, "analysis/candidate_gene_triplets")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

leaf_expr_file <- file.path(input_dir, "leaf_vst_top12000_profile_mean_expression_52profiles.tsv")
peel_expr_file <- file.path(input_dir, "peel_vst_top12000_profile_mean_expression_52profiles.tsv")
trait_file <- file.path(input_dir, "traits_leaf_peel_key_voc_clip0_log1p_52profiles.safe.tsv")

candidate_chain_file <- file.path(enrich_dir, "candidate_module_voc_chains_annotated_with_hubs.tsv")
hub_file <- file.path(enrich_dir, "key_module_voc_related_hub_genes.tsv")

top_n_hubs_per_module <- 20

clean_module <- function(x) {
  x <- gsub("^LeafME__", "", x)
  x <- gsub("^PeelME__", "", x)
  x
}

safe_cor <- function(x, y, method = "spearman", min_n = 20) {
  ok <- is.finite(x) & is.finite(y)
  n_ok <- sum(ok)
  if (n_ok < min_n || sd(x[ok]) == 0 || sd(y[ok]) == 0) {
    return(list(cor = NA_real_, p = NA_real_, n = n_ok))
  }
  ct <- suppressWarnings(cor.test(x[ok], y[ok], method = method, exact = FALSE))
  list(cor = unname(ct$estimate), p = ct$p.value, n = n_ok)
}

residualize_stage <- function(v, stage) {
  if (sd(v, na.rm = TRUE) == 0) return(rep(NA_real_, length(v)))
  residuals(lm(v ~ factor(stage, levels = c("S1", "S2", "S3", "S4"))))
}

leaf_expr <- fread(leaf_expr_file, data.table = FALSE)
peel_expr <- fread(peel_expr_file, data.table = FALSE)
traits <- fread(trait_file, data.table = FALSE)
chains <- fread(candidate_chain_file)
hubs <- fread(hub_file)

# Keep candidate chains from the top-ranked chain output, already annotated
if (!all(c("leaf_module", "peel_module", "peel_voc_trait") %in% colnames(chains))) {
  stop("candidate chain file missing required columns.")
}

leaf_expr <- as.data.table(leaf_expr)
peel_expr <- as.data.table(peel_expr)
traits <- as.data.table(traits)

common_profiles <- Reduce(intersect, list(leaf_expr$group_id, peel_expr$group_id, traits$group_id))
common_profiles <- traits[group_id %in% common_profiles]$group_id

leaf_expr <- leaf_expr[match(common_profiles, group_id)]
peel_expr <- peel_expr[match(common_profiles, group_id)]
traits <- traits[match(common_profiles, group_id)]

if (!all(leaf_expr$group_id == peel_expr$group_id) || !all(leaf_expr$group_id == traits$group_id)) {
  stop("Profile order mismatch.")
}

stage_vec <- traits$stage

leaf_mat <- as.data.frame(leaf_expr[, !"group_id", drop = FALSE])
rownames(leaf_mat) <- leaf_expr$group_id

peel_mat <- as.data.frame(peel_expr[, !"group_id", drop = FALSE])
rownames(peel_mat) <- peel_expr$group_id

trait_mat <- as.data.frame(traits[, !"group_id", drop = FALSE])
rownames(trait_mat) <- traits$group_id

# Hub preparation
if (!all(c("organ", "module", "Geneid", "kME_own_module") %in% colnames(hubs))) {
  stop("hub file missing required columns.")
}

hubs[, abs_kME_own_module := abs(as.numeric(kME_own_module))]
setorder(hubs, organ, module, -abs_kME_own_module)

get_top_hubs <- function(org, mod, available_genes) {
  x <- hubs[organ == org & module == mod & Geneid %in% available_genes]
  if (nrow(x) == 0) return(character(0))
  unique(head(x$Geneid, top_n_hubs_per_module))
}

# Deduplicate chains by leaf module, peel module, peel VOC
chains[, leaf_module_color := clean_module(leaf_module)]
chains[, peel_module_color := clean_module(peel_module)]

chain_cols <- c(
  "leaf_module", "peel_module", "peel_voc_trait",
  "leaf_module_color", "peel_module_color",
  "leaf_peel_module_cor", "leaf_peel_module_fdr",
  "leaf_module_peel_voc_cor", "leaf_module_peel_voc_fdr",
  "peel_module_peel_voc_cor", "peel_module_peel_voc_fdr",
  "peel_voc_original"
)

chain_cols <- intersect(chain_cols, colnames(chains))
chains2 <- unique(chains[, ..chain_cols])

rows <- list()
idx <- 1

for (i in seq_len(nrow(chains2))) {
  ch <- chains2[i]

  leaf_mod <- ch$leaf_module_color
  peel_mod <- ch$peel_module_color
  voc_trait <- ch$peel_voc_trait

  if (!voc_trait %in% colnames(trait_mat)) next

  leaf_genes <- get_top_hubs("Leaf", leaf_mod, colnames(leaf_mat))
  peel_genes <- get_top_hubs("Peel", peel_mod, colnames(peel_mat))

  if (length(leaf_genes) == 0 || length(peel_genes) == 0) next

  voc <- as.numeric(trait_mat[[voc_trait]])

  voc_resid <- residualize_stage(voc, stage_vec)

  for (lg in leaf_genes) {
    lg_expr <- as.numeric(leaf_mat[[lg]])
    lg_resid <- residualize_stage(lg_expr, stage_vec)

    lg_voc <- safe_cor(lg_expr, voc, method = "spearman")
    lg_voc_resid <- safe_cor(lg_resid, voc_resid, method = "pearson")

    for (pg in peel_genes) {
      pg_expr <- as.numeric(peel_mat[[pg]])
      pg_resid <- residualize_stage(pg_expr, stage_vec)

      lg_pg <- safe_cor(lg_expr, pg_expr, method = "spearman")
      pg_voc <- safe_cor(pg_expr, voc, method = "spearman")

      lg_pg_resid <- safe_cor(lg_resid, pg_resid, method = "pearson")
      pg_voc_resid <- safe_cor(pg_resid, voc_resid, method = "pearson")

      direction_gene_coherent <- sign(lg_voc$cor) == sign(lg_pg$cor * pg_voc$cor)

      module_consistent <- TRUE
      if ("leaf_peel_module_cor" %in% colnames(ch)) {
        module_consistent <- module_consistent &&
          sign(lg_pg$cor) == sign(as.numeric(ch$leaf_peel_module_cor))
      }
      if ("leaf_module_peel_voc_cor" %in% colnames(ch)) {
        module_consistent <- module_consistent &&
          sign(lg_voc$cor) == sign(as.numeric(ch$leaf_module_peel_voc_cor))
      }
      if ("peel_module_peel_voc_cor" %in% colnames(ch)) {
        module_consistent <- module_consistent &&
          sign(pg_voc$cor) == sign(as.numeric(ch$peel_module_peel_voc_cor))
      }

      evidence_score <- abs(lg_pg$cor) * abs(lg_voc$cor) * abs(pg_voc$cor)

      rows[[idx]] <- data.table(
        leaf_module = ch$leaf_module,
        peel_module = ch$peel_module,
        peel_voc_trait = voc_trait,
        peel_voc_original = ifelse("peel_voc_original" %in% colnames(ch), ch$peel_voc_original, voc_trait),

        leaf_gene = lg,
        peel_gene = pg,

        leaf_gene_peel_gene_cor = lg_pg$cor,
        leaf_gene_peel_gene_p = lg_pg$p,
        leaf_gene_peel_gene_stage_resid_cor = lg_pg_resid$cor,
        leaf_gene_peel_gene_stage_resid_p = lg_pg_resid$p,

        leaf_gene_peel_voc_cor = lg_voc$cor,
        leaf_gene_peel_voc_p = lg_voc$p,
        leaf_gene_peel_voc_stage_resid_cor = lg_voc_resid$cor,
        leaf_gene_peel_voc_stage_resid_p = lg_voc_resid$p,

        peel_gene_peel_voc_cor = pg_voc$cor,
        peel_gene_peel_voc_p = pg_voc$p,
        peel_gene_peel_voc_stage_resid_cor = pg_voc_resid$cor,
        peel_gene_peel_voc_stage_resid_p = pg_voc_resid$p,

        direction_gene_coherent = direction_gene_coherent,
        module_direction_consistent = module_consistent,
        evidence_score = evidence_score
      )

      idx <- idx + 1
    }
  }
}

triplets <- rbindlist(rows, fill = TRUE)

# Remove duplicated triplets caused by repeated chains
triplets <- unique(triplets)

triplets[, leaf_gene_peel_gene_fdr := p.adjust(leaf_gene_peel_gene_p, method = "BH")]
triplets[, leaf_gene_peel_voc_fdr := p.adjust(leaf_gene_peel_voc_p, method = "BH")]
triplets[, peel_gene_peel_voc_fdr := p.adjust(peel_gene_peel_voc_p, method = "BH")]

setorder(
  triplets,
  -direction_gene_coherent,
  -module_direction_consistent,
  leaf_gene_peel_gene_fdr,
  leaf_gene_peel_voc_fdr,
  peel_gene_peel_voc_fdr,
  -evidence_score
)

fwrite(triplets, file.path(out_dir, "candidate_leaf_gene_peel_gene_peel_voc_triplets_all.tsv"), sep = "\t")

top_triplets <- triplets[
  direction_gene_coherent == TRUE &
    module_direction_consistent == TRUE &
    abs(leaf_gene_peel_gene_cor) >= 0.50 &
    abs(leaf_gene_peel_voc_cor) >= 0.45 &
    abs(peel_gene_peel_voc_cor) >= 0.45
]

setorder(top_triplets, -evidence_score)

fwrite(top_triplets, file.path(out_dir, "candidate_leaf_gene_peel_gene_peel_voc_triplets_top.tsv"), sep = "\t")

# A smaller reviewer-ready table: keep top 10 per VOC
reviewer_ready <- top_triplets[
  ,
  head(.SD, 10),
  by = .(peel_voc_original)
]

setorder(reviewer_ready, peel_voc_original, -evidence_score)

fwrite(reviewer_ready, file.path(out_dir, "candidate_gene_triplets_reviewer_ready_top10_per_voc.tsv"), sep = "\t")

summary <- data.table(
  item = c(
    "n_candidate_module_chains",
    "n_triplets_all",
    "n_triplets_top",
    "n_vocs_with_top_triplets",
    "top_n_hubs_per_module"
  ),
  value = c(
    nrow(chains2),
    nrow(triplets),
    nrow(top_triplets),
    length(unique(top_triplets$peel_voc_original)),
    top_n_hubs_per_module
  )
)

fwrite(summary, file.path(out_dir, "candidate_gene_triplet_summary.tsv"), sep = "\t")

message("Summary:")
print(summary)

message("\nTop triplets:")
print(head(top_triplets, 30))

message("\nDone.")
