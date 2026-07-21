# Supplementary Figures 3–4 transcriptome–VOC support workflow

This folder contains the complete, sanitized workflow used for the supportive leaf–peel transcriptome/VOC analyses underlying Supplementary Figures 3 and 4.

## Scope

- Supplementary Figure 3: global transcriptomic coordination, PCA/PERMANOVA summaries, and pathway-level leaf–peel coordination.
- Supplementary Figure 4: WGCNA module coordination, module–VOC associations, candidate module–VOC chains, and representative hub-gene triplets.

The workflow is grouped here because Supplementary Figures 3 and 4 are generated from the same transcriptome–VOC support analysis. These scripts are not shared with Supplementary Figures 1–2.

## Script order

1. `01_prepare_voc_trait_matrices_clip0.py`
2. `02_prepare_wgcna_inputs.py`
3. `03_run_wgcna_leaf_peel.R`
4. `04_module_trait_association_robust.R`
5. `05_module_enrichment_and_hub_genes.R`
6. `06_candidate_gene_triplets.R`
7. `07_annotate_and_rank_candidate_gene_triplets_no_collision.R`
8. `08_build_final_evidence_package.py`
9. `09_prepare_final_figures_and_tables.R`
10. `10_draw_supplementary_transcriptome_figures.py`
11. `11_draw_supplementary_figure_3.py`
12. `12_draw_supplementary_figure_4.py`

Processed support tables for Supplementary Figures 3 and 4 are provided in the Zenodo data repository under `processed/figure_support/02_suppfig3_4_transcriptome_voc_support/`.
