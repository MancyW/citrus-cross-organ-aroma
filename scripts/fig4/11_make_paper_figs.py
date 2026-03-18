import argparse

from pathlib import Path

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

def pick_latest(glob_pat: str):

    paths = list(Path(".").glob(glob_pat))

    if not paths:

        return None

    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return paths[0]

def ensure_dir(p: Path):

    p.mkdir(parents=True, exist_ok=True)

def load_rank_csv(run_dir: Path, out_tag: str, space: str, weight_mode: str, fp_lam: float):

    fp_tag = f"fpPen{fp_lam:.2f}"

    fname = f"ideotype_ranking_v3.{out_tag}.{fp_tag}.{space}.{weight_mode}.csv"

    f = run_dir / "ideotype_v3" / fname

    if not f.exists():

        fname2 = f"ideotype_ranking_v3.{out_tag}.{space}.{weight_mode}.csv"

        f2 = run_dir / "ideotype_v3" / fname2

        if f2.exists():

            return pd.read_csv(f2), f2

        raise FileNotFoundError(f"Cannot find ranking csv:\n- {f}\n- {f2}")

    return pd.read_csv(f), f

def topk_overlap(true_rank, pred_rank, k):

    true_top = list(true_rank.head(k))

    pred_top = list(pred_rank.head(k))

    inter = len(set(true_top).intersection(set(pred_top)))

    return inter, true_top, pred_top

def plot_panel_sweep(panel_metrics_csv: Path, out_dir: Path):

    df = pd.read_csv(panel_metrics_csv)

    panel_col = "panel" if "panel" in df.columns else df.columns[0]

    cos_candidates = ["mean_cos", "mean_cosine", "mean_cos_sim", "mean_cosine_sim"]

    corr_candidates = ["mean_r", "mean_spearman", "mean_pearson", "mean_corr", "mean_rho"]

    mean_cos = next((c for c in cos_candidates if c in df.columns), None)

    mean_corr = next((c for c in corr_candidates if c in df.columns), None)

    if mean_cos is None:

        raise ValueError(

            f"panel_metrics.csv missing cosine column. "

            f"Tried={cos_candidates}, got={df.columns.tolist()}"

        )

    df = df.sort_values(mean_cos, ascending=False).copy()

    fig = plt.figure(figsize=(8, 4.5))

    x = np.arange(len(df))

    plt.plot(x, df[mean_cos].values, marker="o", label=mean_cos)

    if mean_corr is not None:

        plt.plot(x, df[mean_corr].values, marker="o", label=mean_corr)

    plt.xticks(x, df[panel_col].astype(str).values, rotation=30, ha="right")

    plt.ylabel("score")

    plt.title(f"Panel sweep (from {panel_metrics_csv.parent.name})")

    plt.legend()

    plt.tight_layout()

    out_png = out_dir / "fig_panel_sweep.png"

    out_pdf = out_dir / "fig_panel_sweep.pdf"

    fig.savefig(out_png, dpi=300)

    fig.savefig(out_pdf)

    plt.close(fig)

    return out_png, out_pdf

def plot_ideotype_scatter(rank: pd.DataFrame, anchor: str, out_dir: Path, tag: str):

    ycol = "pred_sim_pen" if "pred_sim_pen" in rank.columns else "pred_sim"

    if "true_sim" not in rank.columns:

        raise ValueError("ranking csv missing true_sim")

    if "cultivar" not in rank.columns:

        raise ValueError("ranking csv missing cultivar")

    fig = plt.figure(figsize=(6, 5))

    x = rank["true_sim"].astype(float).values

    y = rank[ycol].astype(float).values

    plt.scatter(x, y)

    for _, r in rank.iterrows():

        c = str(r["cultivar"])

        dx, dy = 0.002, 0.002

        if c == anchor:

            plt.annotate(c, (float(r["true_sim"]), float(r[ycol])), fontsize=10, fontweight="bold")

        else:

            plt.annotate(c, (float(r["true_sim"])+dx, float(r[ycol])+dy), fontsize=8)

    plt.xlabel("true similarity")

    plt.ylabel(ycol)

    plt.title(f"Ideotype similarity: {tag}")

    plt.tight_layout()

    out_png = out_dir / f"fig_ideotype_scatter.{tag}.png"

    out_pdf = out_dir / f"fig_ideotype_scatter.{tag}.pdf"

    fig.savefig(out_png, dpi=300)

    fig.savefig(out_pdf)

    plt.close(fig)

    return out_png, out_pdf

def plot_topk(rank: pd.DataFrame, anchor: str, out_dir: Path, tag: str):

    ycol = "pred_sim_pen" if "pred_sim_pen" in rank.columns else "pred_sim"

    pred_rank = rank.sort_values(ycol, ascending=False)

    true_rank = rank.sort_values("true_sim", ascending=False)

    inc = {}

    for k in [1, 3, 5]:

        inter, _, _ = topk_overlap(true_rank["cultivar"], pred_rank["cultivar"], k)

        inc[k] = inter / k

    pred_ex = pred_rank[pred_rank["cultivar"] != anchor]

    true_ex = true_rank[true_rank["cultivar"] != anchor]

    exc = {}

    for k in [1, 3, 5]:

        inter, _, _ = topk_overlap(true_ex["cultivar"], pred_ex["cultivar"], k)

        exc[k] = inter / k

    fig = plt.figure(figsize=(6.5, 4))

    ks = [1, 3, 5]

    x = np.arange(len(ks))

    width = 0.35

    plt.bar(x - width/2, [inc[k] for k in ks], width, label="include anchor")

    plt.bar(x + width/2, [exc[k] for k in ks], width, label="exclude anchor")

    plt.xticks(x, [f"top{k}" for k in ks])

    plt.ylim(0, 1.05)

    plt.ylabel("overlap fraction")

    plt.title(f"TopK overlap: {tag}")

    plt.legend()

    plt.tight_layout()

    out_png = out_dir / f"fig_topk_overlap.{tag}.png"

    out_pdf = out_dir / f"fig_topk_overlap.{tag}.pdf"

    fig.savefig(out_png, dpi=300)

    fig.savefig(out_pdf)

    plt.close(fig)

    return out_png, out_pdf

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--main_run_dir", type=str, default="", help="Main figure run_dir (recommend: LOCO S1+S4)")

    ap.add_argument("--supp_run_dirs", type=str, nargs="*", default=[], help="Optional extra run_dirs for supplementary figs")

    ap.add_argument("--anchor", type=str, default="MTH")

    ap.add_argument("--out_tag", type=str, default="weighted_cosine.f1rho.thresh")

    ap.add_argument("--space", type=str, default="absolute_log1p")

    ap.add_argument("--weight_mode", type=str, default="rhof1")

    ap.add_argument("--fp_penalty_lambda", type=float, default=0.2)

    ap.add_argument("--out_dir", type=str, default="results/paper_figs")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    ensure_dir(out_dir)

    panel_csv = pick_latest("results/runs/run_*/panel_metrics.csv")

    if panel_csv is not None:

        png, pdf = plot_panel_sweep(panel_csv, out_dir)

        print(f"[OK] panel sweep fig: {png} | {pdf}")

    else:

        print("[WARN] cannot find results/runs/run_*/panel_metrics.csv, skip panel sweep fig")

    main_run = Path(args.main_run_dir) if args.main_run_dir else None

    if main_run is None or not main_run.exists():

        root = Path("results/pred_vectors")

        cands = sorted([p for p in root.glob("run_*_S1_S4") if p.is_dir() and "FITALL" not in p.name],

                       key=lambda p: p.stat().st_mtime, reverse=True)

        if cands:

            main_run = cands[0]

            print(f"[Auto] main_run_dir={main_run}")

        else:

            raise FileNotFoundError("No main_run_dir provided and cannot auto-detect LOCO S1+S4 run_*_S1_S4")

    rank, rank_path = load_rank_csv(main_run, args.out_tag, args.space, args.weight_mode, args.fp_penalty_lambda)

    tag = f"MAIN.{main_run.name}"

    plot_ideotype_scatter(rank, args.anchor, out_dir, tag)

    plot_topk(rank, args.anchor, out_dir, tag)

    print(f"[OK] main figs from: {rank_path}")

    for rd in args.supp_run_dirs:

        run_dir = Path(rd)

        rank, rank_path = load_rank_csv(run_dir, args.out_tag, args.space, args.weight_mode, args.fp_penalty_lambda)

        tag = f"SUPP.{run_dir.name}"

        plot_ideotype_scatter(rank, args.anchor, out_dir, tag)

        plot_topk(rank, args.anchor, out_dir, tag)

        print(f"[OK] supp figs from: {rank_path}")

    print(f"[DONE] figures saved to: {out_dir}")

if __name__ == "__main__":

    main()
