import argparse

import shutil

from pathlib import Path

import pandas as pd

from datetime import datetime

def copy_if_exists(src: Path, dst: Path, manifest_rows: list, kind: str, note: str = ""):

    if src.exists():

        dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src, dst)

        manifest_rows.append({

            "kind": kind,

            "src": str(src),

            "dst": str(dst),

            "note": note

        })

        return True

    return False

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--out_tag", default="weighted_cosine.f1rho.thresh")

    ap.add_argument("--space", default="absolute_log1p")

    ap.add_argument("--weight_mode", default="rhof1")

    ap.add_argument("--fp_penalty_lambda", type=float, default=0.2)

    ap.add_argument("--anchor", default="MTH")

    ap.add_argument("--main_run_dir", required=True)

    ap.add_argument("--supp_run_dirs", nargs="+", required=True)

    ap.add_argument("--src_fig_dir", default="results/paper_figs")

    ap.add_argument("--src_tab_dir", default="results/paper_tables")

    ap.add_argument("--dst_root", default="results/paper_package")

    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    tag = f"{args.out_tag}.fpPen{args.fp_penalty_lambda:.2f}.{args.space}.{args.weight_mode}"

    out_dir = Path(args.dst_root) / f"{stamp}.{tag}"

    (out_dir / "figs").mkdir(parents=True, exist_ok=True)

    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    (out_dir / "configs").mkdir(parents=True, exist_ok=True)

    fig_dir = Path(args.src_fig_dir)

    tab_dir = Path(args.src_tab_dir)

    manifest = []

    table_files = [

        tab_dir / f"main_table_penalty_fpPen{args.fp_penalty_lambda:.2f}.ideotype_topk.csv",

        tab_dir / f"supplement_table_penalty_fpPen{args.fp_penalty_lambda:.2f}.ideotype_topk.csv",

        tab_dir / f"robust_bootstrap_ci.{args.space}.{args.weight_mode}.fpPen{args.fp_penalty_lambda:.2f}.csv",

        tab_dir / f"robust_manifest.{args.space}.{args.weight_mode}.fpPen{args.fp_penalty_lambda:.2f}.csv",

        tab_dir / f"ablation_best.{args.out_tag}.fpPen{args.fp_penalty_lambda:.2f}.csv",

    ]

    run_names = [Path(args.main_run_dir).name] + [Path(x).name for x in args.supp_run_dirs]

    for rn in run_names:

        table_files.append(tab_dir / f"robust_lambda_curve.{rn}.{args.space}.{args.weight_mode}.csv")

        table_files.append(tab_dir / f"ablation_grid.{rn}.{args.out_tag}.fpPen{args.fp_penalty_lambda:.2f}.csv")

    for src in table_files:

        dst = out_dir / "tables" / src.name

        copy_if_exists(src, dst, manifest, kind="table")

    patterns = [

        "fig_panel_sweep.*",

        "fig_ideotype_scatter.*.png", "fig_ideotype_scatter.*.pdf",

        "fig_topk_overlap.*.png", "fig_topk_overlap.*.pdf",

        "fig_lambda_curve.*.png", "fig_lambda_curve.*.pdf",

        "fig_fp_rate.*.png", "fig_fp_rate.*.pdf",

        "fig_ablation_heatmap.*.png", "fig_ablation_heatmap.*.pdf",

    ]

    for pat in patterns:

        for src in sorted(fig_dir.glob(pat)):

            dst = out_dir / "figs" / src.name

            copy_if_exists(src, dst, manifest, kind="figure")

    cfgs = [Path("configs/fig4/base.yaml"), Path("configs/fig4/fitall.yaml")]

    for src in cfgs:

        dst = out_dir / "configs" / src.name

        copy_if_exists(src, dst, manifest, kind="config")

    man_df = pd.DataFrame(manifest)

    man_path = out_dir / "manifest.csv"

    man_df.to_csv(man_path, index=False)

    readme = []

    readme.append("# Paper package\n")

    readme.append(f"- tag: {tag}\n")

    readme.append(f"- anchor: {args.anchor}\n")

    readme.append(f"- main_run_dir: {args.main_run_dir}\n")

    readme.append(f"- supp_run_dirs: {', '.join(args.supp_run_dirs)}\n\n")

    ci_path = tab_dir / f"robust_bootstrap_ci.{args.space}.{args.weight_mode}.fpPen{args.fp_penalty_lambda:.2f}.csv"

    if ci_path.exists():

        df = pd.read_csv(ci_path)

        readme.append("## Robustness (bootstrap CI)\n\n")

        for _, r in df.iterrows():

            readme.append(

                f"- {r['run_name']}: rho={r['rho']:.6f}, 95%CI=[{r['ci_lo']:.6f}, {r['ci_hi']:.6f}], n={int(r['n'])}\n"

            )

        readme.append("\n")

    best_path = tab_dir / f"ablation_best.{args.out_tag}.fpPen{args.fp_penalty_lambda:.2f}.csv"

    if best_path.exists():

        dfb = pd.read_csv(best_path)

        readme.append("## Ablation best per run\n\n")

        for _, r in dfb.iterrows():

            readme.append(

                f"- {r['run_name']}: space={r['space']}, weight={r['weight_mode']}, "

                f"calib={bool(r['calibrate'])}, spearman_excl={r['spearman_exclude_anchor']:.6f}, "

                f"top3={r['top3_exclude_anchor_pred']}\n"

            )

        readme.append("\n")

    (out_dir / "README.md").write_text("".join(readme), encoding="utf-8")

    print("[DONE] paper package dir:", out_dir)

    print("[OK] wrote:", man_path)

    print("[OK] wrote:", out_dir / "README.md")

if __name__ == "__main__":

    main()
