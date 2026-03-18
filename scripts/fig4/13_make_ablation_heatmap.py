import argparse

import re

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import spearmanr

SPACE_CANDIDATES = ["absolute_log1p", "clr", "zscore_log1p"]

WEIGHT_CANDIDATES = ["rhof1", "rho"]

def _safe_spearman(a, b):

    a = np.asarray(a, dtype=float)

    b = np.asarray(b, dtype=float)

    if np.all(a == a[0]) or np.all(b == b[0]):

        return np.nan

    return spearmanr(a, b).correlation

def parse_meta_from_filename(name: str):

    meta = {

        "space": None,

        "weight_mode": None,

        "calibrate": False,

        "fp_penalty_lambda": None,

    }

    for sp in SPACE_CANDIDATES:

        if sp in name:

            meta["space"] = sp

            break

    for wm in WEIGHT_CANDIDATES:

        if re.search(rf"(?<![A-Za-z0-9]){re.escape(wm)}(?![A-Za-z0-9])", name) or f".{wm}" in name:

            meta["weight_mode"] = wm

            break

    if "rhof1" in name:

        meta["weight_mode"] = "rhof1"

    elif "rho" in name and meta["weight_mode"] is None:

        meta["weight_mode"] = "rho"

    meta["calibrate"] = ("calib" in name.lower())

    m = re.search(r"fpPen([0-9]+\.[0-9]+)", name)

    if m:

        meta["fp_penalty_lambda"] = float(m.group(1))

    else:

        m2 = re.search(r"fpPen([0-9]+\.[0-9]+|[0-9]+)", name)

        if m2:

            try:

                meta["fp_penalty_lambda"] = float(m2.group(1))

            except Exception:

                pass

    return meta

def scan_ranking_files(run_dir: Path, out_tag: str, fp_penalty_lambda: float):

    ideotype_dir = run_dir / "ideotype_v3"

    if not ideotype_dir.exists():

        raise FileNotFoundError(f"Missing: {ideotype_dir}")

    files = sorted(ideotype_dir.glob(f"ideotype_ranking_v3.{out_tag}*.csv"))

    if not files:

        raise FileNotFoundError(f"No ranking csv found under {ideotype_dir} with out_tag={out_tag}")

    fp_token = f"fpPen{fp_penalty_lambda:.2f}"

    files = [p for p in files if fp_token in p.name]

    if not files:

        raise FileNotFoundError(

            f"No ranking csv with token {fp_token} under {ideotype_dir}. "

            f"Found {len(list((run_dir/'ideotype_v3').glob('ideotype_ranking_v3.*.csv')))} csv total."

        )

    return files

def compute_metrics(rank_csv: Path, anchor: str, score_col_prefer: str = "pred_sim_pen"):

    df = pd.read_csv(rank_csv)

    if score_col_prefer in df.columns:

        score_col = score_col_prefer

    elif "pred_sim" in df.columns:

        score_col = "pred_sim"

    else:

        raise ValueError(f"{rank_csv.name} has no pred_sim_pen or pred_sim. cols={df.columns.tolist()}")

    if "true_sim" not in df.columns:

        raise ValueError(f"{rank_csv.name} missing true_sim column. cols={df.columns.tolist()}")

    rho_incl = _safe_spearman(df[score_col], df["true_sim"])

    df_no = df[df["cultivar"] != anchor].copy()

    rho_excl = _safe_spearman(df_no[score_col], df_no["true_sim"])

    top3 = ",".join(df_no.sort_values(score_col, ascending=False).head(3)["cultivar"].tolist())

    top5 = ",".join(df_no.sort_values(score_col, ascending=False).head(5)["cultivar"].tolist())

    return {

        "score_col": score_col,

        "spearman_include_anchor": rho_incl,

        "spearman_exclude_anchor": rho_excl,

        "top3_exclude_anchor_pred": top3,

        "top5_exclude_anchor_pred": top5,

    }

def plot_heatmap(grid_df: pd.DataFrame, out_png: Path, out_pdf: Path, title: str):

    spaces = [sp for sp in SPACE_CANDIDATES if sp in set(grid_df["space"])]

    cols = []

    for wm in ["rho", "rhof1"]:

        for cal in [False, True]:

            cols.append((wm, cal))

    mat = np.full((len(spaces), len(cols)), np.nan)

    for i, sp in enumerate(spaces):

        for j, (wm, cal) in enumerate(cols):

            sub = grid_df[(grid_df["space"] == sp) & (grid_df["weight_mode"] == wm) & (grid_df["calibrate"] == cal)]

            if len(sub) > 0:

                mat[i, j] = float(sub["spearman_exclude_anchor"].iloc[0])

    fig = plt.figure(figsize=(10, 3.6))

    ax = plt.gca()

    im = ax.imshow(mat, aspect="auto", vmin=np.nanmin(mat), vmax=np.nanmax(mat))

    ax.set_yticks(range(len(spaces)))

    ax.set_yticklabels(spaces)

    ax.set_xticks(range(len(cols)))

    ax.set_xticklabels([f"{wm}\ncalib={int(cal)}" for wm, cal in cols])

    for i in range(mat.shape[0]):

        for j in range(mat.shape[1]):

            v = mat[i, j]

            if np.isfinite(v):

                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9)

    ax.set_title(title)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Spearman (exclude anchor)")

    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_png, dpi=250)

    fig.savefig(out_pdf)

    plt.close(fig)

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--run_dirs", nargs="+", required=True)

    ap.add_argument("--anchor", default="MTH")

    ap.add_argument("--out_tag", default="weighted_cosine.f1rho.thresh")

    ap.add_argument("--fp_penalty_lambda", type=float, default=0.2)

    ap.add_argument("--paper_fig_dir", default="results/paper_figs")

    ap.add_argument("--paper_table_dir", default="results/paper_tables")

    args = ap.parse_args()

    fig_dir = Path(args.paper_fig_dir)

    tab_dir = Path(args.paper_table_dir)

    fig_dir.mkdir(parents=True, exist_ok=True)

    tab_dir.mkdir(parents=True, exist_ok=True)

    best_rows = []

    for rd in args.run_dirs:

        run_dir = Path(rd)

        run_name = run_dir.name

        files = scan_ranking_files(run_dir, args.out_tag, args.fp_penalty_lambda)

        rows = []

        for p in files:

            meta = parse_meta_from_filename(p.name)

            if meta["space"] is None or meta["weight_mode"] is None:

                continue

            met = compute_metrics(p, anchor=args.anchor, score_col_prefer="pred_sim_pen")

            row = {

                "run_name": run_name,

                "rank_csv": str(p),

                "space": meta["space"],

                "weight_mode": meta["weight_mode"],

                "calibrate": bool(meta["calibrate"]),

                "fp_penalty_lambda": args.fp_penalty_lambda,

                **met,

            }

            rows.append(row)

        if not rows:

            raise RuntimeError(

                f"No parsable ranking csv for run={run_name}. "

                f"Check naming contains space/weight_mode/calib and fpPen{args.fp_penalty_lambda:.2f}."

            )

        df = pd.DataFrame(rows)

        df = df.sort_values("spearman_exclude_anchor", ascending=False)

        out_grid = tab_dir / f"ablation_grid.{run_name}.{args.out_tag}.fpPen{args.fp_penalty_lambda:.2f}.csv"

        df.to_csv(out_grid, index=False)

        best = df.iloc[0].to_dict()

        best_rows.append(best)

        grid_unique = (

            df.sort_values("spearman_exclude_anchor", ascending=False)

              .drop_duplicates(subset=["space", "weight_mode", "calibrate"], keep="first")

        )

        title = f"Ablation heatmap: {run_name} | fpPen={args.fp_penalty_lambda:.2f}"

        out_png = fig_dir / f"fig_ablation_heatmap.{run_name}.{args.out_tag}.fpPen{args.fp_penalty_lambda:.2f}.png"

        out_pdf = fig_dir / f"fig_ablation_heatmap.{run_name}.{args.out_tag}.fpPen{args.fp_penalty_lambda:.2f}.pdf"

        plot_heatmap(grid_unique, out_png, out_pdf, title)

        print(f"[OK] {run_name}")

        print(f"  - grid : {out_grid}")

        print(f"  - heat : {out_png} | {out_pdf}")

        print(f"  - best : space={best['space']} weight={best['weight_mode']} calib={best['calibrate']} "

              f"spearman_excl={best['spearman_exclude_anchor']:.4f} top3={best['top3_exclude_anchor_pred']}")

    best_df = pd.DataFrame(best_rows).sort_values(["run_name", "spearman_exclude_anchor"], ascending=[True, False])

    out_best = tab_dir / f"ablation_best.{args.out_tag}.fpPen{args.fp_penalty_lambda:.2f}.csv"

    best_df.to_csv(out_best, index=False)

    print(f"\n[OK] wrote best summary: {out_best}")

if __name__ == "__main__":

    main()
