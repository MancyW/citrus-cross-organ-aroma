# A cross-organ proxy-phenotyping framework for inferring mature peel volatile phenotypes from leaf volatiles in citrus

## Overview
This repository contains the curated code used to generate the main analyses and figure panels for the manuscript:

**A cross-organ proxy-phenotyping framework for inferring mature peel volatile phenotypes from leaf volatiles in citrus**

The study integrates organ-resolved VOC profiling, cross-organ covariance analysis, cultivar-level proxy inference, and model-derived odor-semantic analysis to test whether leaf volatile profiles carry information about mature peel volatile phenotypes in citrus.

This repository is intentionally code-focused. The processed data package and journal Source Data files are deposited separately in Zenodo.

## Repository scope
The repository contains code supporting six main figures and four supplementary figures:

- **Fig. 1** — Peel VOC profiling across developmental stages
- **Fig. 2** — Leaf VOC profiling across developmental stages
- **Fig. 3** — Cross-organ ordination, overlap, covariance, and module-level coupling
- **Fig. 4** — Stage-selective proxy inference framework and cultivar ranking
- **Fig. 5** — Model benchmarking, interpretability, and applicability diagnostics
- **Fig. 6** — Odor-semantic embedding, descriptor shift, retrieval, and robustness
- **Supplementary Fig. 1** — Peel-S4 chemotype-blocked extrapolation stress test
- **Supplementary Fig. 2** — Original-space validation of cross-organ odor-semantic transfer
- **Supplementary Fig. 3** — Cross-organ transcriptomic coordination between leaf and peel
- **Supplementary Fig. 4** — Module-level transcriptome–VOC associations

The current release is a curated code archive. It retains the core analysis scripts and minimal supporting configuration files required to understand and reproduce the main computational workflow.

## Data availability for code execution
This repository does **not** bundle the full processed data package or raw instrument files.

To run the code, place the processed data files from the Zenodo data release into the expected local `data/` directory structure.

### External data package
Processed data and Source Data files are available at:

- **Zenodo archive DOI:** `10.5281/zenodo.21476479`

### Journal-facing Source Data
The final figure-linked Source Data workbooks are deposited separately and correspond to:
- `SourceData_Fig1.xlsx`
- `SourceData_Fig2.xlsx`
- `SourceData_Fig3.xlsx`
- `SourceData_Fig4.xlsx`
- `SourceData_Fig5.xlsx`
- `SourceData_Fig6.xlsx`

### Raw GC–MS vendor files
Raw GC–MS vendor files are **not included** in this repository. As stated in the manuscript, raw GC–MS files and additional intermediate files are available from the corresponding author upon reasonable request.

## Repository structure
```text
configs/
  fig4/
    base.yaml
    fitall.yaml

scripts/
  fig1/
  fig2/
  fig3/
  fig4/
  fig5/
  fig6/
  suppfig/

src/
  fig4/
    cv/
    models/
    run/
    ssot/
    utils/
  fig5_common/

```

### `configs/fig4/`
Configuration files used in the Fig. 4 proxy-inference workflow.

### `scripts/fig1/`
Panel-level scripts for peel VOC analysis, including representative TIC display, PLS-DA, heatmap generation, family-level summaries, marker selection, clustering, and marker-specific matrix panels.

### `scripts/fig2/`
Panel-level scripts for leaf VOC analysis, including representative TIC display, PLS-DA/QC, heatmap generation, family-level summaries, marker selection, clustering, and marker-specific matrix panels.

### `scripts/fig3/`
Scripts for cross-organ ordination, VOC overlap, covariance network construction, representative pair analysis, and Sankey-style module/axis summaries.

### `scripts/fig4/`
Shell and Python scripts for the cultivar-level proxy-inference workflow, including SSOT construction, panel sweep, final model execution, robustness analysis, paired bootstrap comparison, and paper-figure generation.

### `scripts/fig5/`
Scripts for baseline benchmarking, interpretability analysis, dynamics analysis, failure / applicability analysis, bootstrap comparison, and permutation-based null evaluation.

### `scripts/fig6/`
Scripts for odor-semantic analysis, including input preparation, OpenPOM-style training or inference, descriptor-probability construction, QC filtering, retrieval evaluation, robustness analysis, and final figure assembly.

### `scripts/suppfig/`
Scripts and documentation supporting Supplementary Figs. 1–4. Supplementary Figs. 1 and 2 are grouped as a shared model-validation and odor-semantic support module, whereas Supplementary Figs. 3 and 4 are grouped as a shared transcriptome–VOC support module.

### `src/fig4/`
Reusable modules for Fig. 4, including:
- cross-validation
- hurdle-model logic
- SSOT construction
- panel-sweep and ranking utilities
- post-processing and evaluation helpers

### `src/fig5_common/`
Reusable helper functions used by the Fig. 5 module family, including:
- data loading
- weights parsing
- statistics
- plotting helpers
- path handling
- association building

## Expected local directories
Many scripts assume a project-style local structure in which data and outputs are stored separately from code. The exact paths vary by module, but the code generally expects one or more of the following directories to exist:

```text
data/
results/
intermediate/
models/
output/
```

At minimum, users should check the relevant script before execution and provide the required input and output paths through the documented configuration files or command-line arguments for their local environment.

## Figure-specific entry points

### Fig. 1
Main scripts in `scripts/fig1/`:
- `01_tic_panel.py`
- `02_plsda_panel.py`
- `03_heatmap_panel.py`
- `04_family_total_panel.py`
- `05_marker_selection_panel.py`
- `06_cluster_panel.py`
- `07_marker_matrix_panel.py`

These scripts generate the peel VOC figure components corresponding to representative TIC traces, PLS-DA, stage-wise heatmaps, family summaries, marker screening, clustering, and marker matrices.

### Fig. 2
Main scripts in `scripts/fig2/`:
- `01_tic_panel.py`
- `02_plsda_panel.py`
- `03_heatmap_panel.py`
- `04_family_total_panel.py`
- `05_marker_selection_panel.py`
- `06_cluster_panel.py`
- `07_marker_matrix_panel.py`

These scripts generate the leaf VOC figure components, including QC-aware PLS-DA and marker-support outputs.

### Fig. 3
Main scripts in `scripts/fig3/`:
- `01_ordination_panel.py`
- `02_overlap_panel.py`
- `03_network_panel.py`
- `04_representative_pairs_panel_with_SEM.py`
- `05_sankey_panel.py`

These scripts implement the cross-organ analysis layer, including ordination, repertoire overlap, network analysis, representative pair tracking, and higher-level Sankey-style aggregation.

### Fig. 4
Main scripts in `scripts/fig4/` include:
- `01_build_ssot.sh`
- `02_run_panel_sweep.sh`
- `04_panel_robustness_report.sh`
- `05_summarize_all_panels.py`
- `06_panel_selection_bootstrap.py`
- `07_panel_skill_over_baseline.py`
- `08_compare_two_panels_paired_bootstrap.py`
- `09_run_final_two_panels.sh`
- `10_run_paper_eval.sh`
- `11_make_paper_figs.py`
- `12_make_robustness.py`
- `13_make_ablation_heatmap.py`
- `13_summarize_ideotype_ablations.py`
- `14_pack_paper_assets.py`

Together with `src/fig4/`, these files define the proxy-inference and cultivar-ranking framework used in Fig. 4.

### Fig. 5
Main scripts in `scripts/fig5/` include:
- `01_run_baselines_loco.py`
- `01_interpretability.py`
- `01_dynamics_analysis.py`
- `01_failure_analysis.py`
- `10_bootstrap_ai_vs_bestbaseline.py`
- `11_doa_error_model.py`
- `11_permutation_null.py`
- `run_nc_all.sh`

These scripts implement the baseline comparison, interpretability, dynamic/static decomposition, failure-aware analysis, and null benchmarking reported in Fig. 5.

### Fig. 6
Main scripts in `scripts/fig6/` include:
- `00_prepare_inputs.py`
- `01_train_openpom_cpu.py`
- `02_infer_voc_and_build_sample_vectors.py`
- `04_export_fig6a_region_support_tables.py`
- `05_qc_filter_and_build_relative_inputs.py`
- `06_plot_fig6b_transfer.py`
- `07_plot_fig6d_retrieval.py`
- `08_plot_fig6e_robustness.py`
- `09_plot_fig6c_descriptor_shift.py`
- `10_make_fig6_final_panels.py`

These scripts implement the odor-semantic analysis workflow, from descriptor model preparation to final panel generation.

### Supplementary Figures 1–4

- `01_suppfig1_2_model_and_odor_semantic/` contains the model-validation and original-space odor-semantic support scripts associated with Supplementary Figs. 1 and 2. Processed figure-support tables are provided in the Zenodo archive.
- `02_suppfig3_4_transcriptome_voc_support/` contains transcriptome preprocessing, WGCNA, pathway-level coordination, module–VOC association, candidate-triplet prioritization, and plotting workflows supporting Supplementary Figs. 3 and 4.

Processed inputs and figure-support tables are available from the Zenodo data archive.

## Software environment
The scripts depend on standard scientific Python packages together with several specialized packages used in the semantic-analysis module.

Core packages used across the repository include:
- Python 3
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `scikit-learn`
- `pyyaml`
- `joblib`

Additional packages used in selected modules include:
- `plotly`
- `rdkit`
- `umap-learn`
- `deepchem`
- `openpom`
- `torch` / DGL backend support where required by the OpenPOM workflow

Because this repository is a curated extraction of the original analysis code, users should build the environment according to the needs of the specific figure module they intend to run.

## Reproducibility notes
1. **Data are external to this repository.**  
   The code assumes that processed data are available locally after download from the Zenodo data release.

2. **Run module-based workflows from the repository root.**  
   Reusable Fig. 4 and Fig. 5 modules are included under `src/`. Run the documented module entry points from the repository root so that local packages are resolved consistently. Users may configure input and output paths, but should not need to edit Python import paths.

3. **Figure 4 and Fig. 5 are the most workflow-oriented modules.**  
   These parts are best run as project pipelines rather than as isolated single scripts.

4. **Figure 1–3 scripts are panel-oriented.**  
   These are generally easier to execute as stand-alone plotting/analysis scripts once the corresponding data inputs are available.

5. **Figure 6 includes an external semantic-model component.**  
   Running the full Fig. 6 pipeline may require third-party model dependencies and an external curated odor dataset, depending on whether users wish to retrain the model or only use prepared outputs.

## Suggested usage
A practical order for reuse is:

1. Download the processed data package from Zenodo.
2. Create the expected local `data/` directory.
3. Start with the panel-level scripts for Fig. 1–3.
4. Run the Fig. 4 workflow using the provided configuration files.
5. Run Fig. 5 benchmarking and diagnostics using the selected model outputs.
6. Use the Fig. 6 scripts if semantic-model dependencies are available.

## Contact
For questions about code execution, processed data dependencies, or raw-data access, please contact the corresponding author listed in the manuscript.
