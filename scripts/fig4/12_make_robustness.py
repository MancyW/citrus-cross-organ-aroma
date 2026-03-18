import argparse

from pathlib import Path

import numpy as np

import pandas as pd

from scipy.stats import spearmanr

import matplotlib.pyplot as plt

def safe_spearman(a, b):

    a = np.asarray(a)

    b = np.asarray(b)

    if len(a) < 3:

        return np.nan

    if np.all(a == a[0]) or np.all(b == b[0]):

        return np.nan

    return spearmanr(a, b).correlation

def load_rank_csv(run_dir: Path, out_tag: str, space: str, weight_mode: str, fp_penalty_lambda: float):

    ideotype_dir = run_dir / "ideotype_v3"

    fname = f"ideotype_ranking_v3.{out_tag}.fpPen{fp_penalty_lambda:0.2f}.{space}.{weight_mode}.csv"

    path = ideotype_dir / fname

    if not path.exists():

        raise FileNotFoundError(f"Cannot find ranking csv: {path}")

    return pd.read_csv(path)

def find_thresh_pv(run_dir: Path):

    cands = [

        run_dir / "pred_vectors_long.thresh_q0.05_log1p.parquet",

        run_dir / "pred_vectors_long.thresh_q0.05_raw.parquet",

        run_dir / "pred_vectors_long.thresh_q0.05_log1p.parquet".replace("0.05", "0.05"),

    ]

    for p in cands:

        if p.exists():

            return p

    any_thresh = sorted(run_dir.glob("pred_vectors_long.thresh_*.parquet"))

    if any_thresh:

        return any_thresh[0]

    raise FileNotFoundError(f"Cannot find thresholded pred_vectors parquet under {run_dir}")

def load_pv_thresh(run_dir: Path):

    pv_path = find_thresh_pv(run_dir)

    pv = pd.read_parquet(pv_path)

    return pv, pv_path

def compute_fp_rate(pv: pd.DataFrame) -> pd.DataFrame:

    df = pv.copy()

    df["yt"] = df["y_true"].astype(float)

    df["yp"] = df["y_pred"].astype(float)

    is_true0 = (df["yt"] == 0.0)

    is_fp = is_true0 & (df["yp"] > 0.0)

    df["is_true0"] = is_true0.astype(int)

    df["is_fp"] = is_fp.astype(int)

    agg = df.groupby("cultivar", as_index=False)[["is_true0", "is_fp"]].sum()

    agg["fp_rate"] = agg["is_fp"] / agg["is_true0"].clip(lower=1)

    return agg[["cultivar", "fp_rate"]]

def ensure_fp_rate_col(rank: pd.DataFrame, fp_rate_df: pd.DataFrame) -> pd.DataFrame:

    df = rank.copy()

    if "fp_rate" in df.columns:

        df["fp_rate"] = pd.to_numeric(df["fp_rate"], errors="coerce").fillna(0.0)

        return df

    df = df.merge(fp_rate_df, on="cultivar", how="left", suffixes=("_x", "_y"))

    if "fp_rate" in df.columns:

        df["fp_rate"] = pd.to_numeric(df["fp_rate"], errors="coerce").fillna(0.0)

        return df

    candidates = [c for c in df.columns if c.startswith("fp_rate")]

    if candidates:

        pick = "fp_rate_y" if "fp_rate_y" in candidates else candidates[0]

        df["fp_rate"] = pd.to_numeric(df[pick], errors="coerce").fillna(0.0)

    else:

        df["fp_rate"] = 0.0

    return df

def penalty_curve(rank: pd.DataFrame, fp_rate_df: pd.DataFrame, anchor: str, lambdas, base_lambda: float):

    df = ensure_fp_rate_col(rank, fp_rate_df)

    if "pred_sim" not in df.columns:

        if "pred_sim_pen" in df.columns:

            df["pred_sim"] = pd.to_numeric(df["pred_sim_pen"], errors="coerce").fillna(0.0) + base_lambda * df["fp_rate"]

        else:

            raise ValueError("Ranking CSV has neither 'pred_sim' nor 'pred_sim_pen' columns; cannot do lambda curve.")

    df_no_anchor = df[df["cultivar"] != anchor].copy()

    out = []

    for lam in lambdas:

        df_no_anchor["pred_sim_pen_tmp"] = pd.to_numeric(df_no_anchor["pred_sim"], errors="coerce").fillna(0.0) - lam * df_no_anchor["fp_rate"]

        rho = safe_spearman(df_no_anchor["pred_sim_pen_tmp"], df_no_anchor["true_sim"])

        top3 = ",".join(df_no_anchor.sort_values("pred_sim_pen_tmp", ascending=False).head(3)["cultivar"].tolist())

        top5 = ",".join(df_no_anchor.sort_values("pred_sim_pen_tmp", ascending=False).head(5)["cultivar"].tolist())

        out.append({"lambda": lam, "spearman_exclude_anchor": rho, "top3": top3, "top5": top5})

    return pd.DataFrame(out)

def bootstrap_ci_spearman(rank: pd.DataFrame, anchor: str, score_col: str, n_boot: int = 2000, seed: int = 0):

    rng = np.random.default_rng(seed)

    df = rank[rank["cultivar"] != anchor].copy()

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col, "true_sim"])

    n = len(df)

    if n < 5:

        return {"n": n, "rho": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    base_rho = safe_spearman(df[score_col], df["true_sim"])

    rhos = []

    idx = np.arange(n)

    for _ in range(n_boot):

        samp = rng.choice(idx, size=n, replace=True)

        d = df.iloc[samp]

        r = safe_spearman(d[score_col], d["true_sim"])

        if not np.isnan(r):

            rhos.append(r)

    if len(rhos) < 30:

        return {"n": n, "rho": float(base_rho), "ci_lo": np.nan, "ci_hi": np.nan}

    rhos = np.sort(np.array(rhos))

    ci_lo = float(np.quantile(rhos, 0.025))

    ci_hi = float(np.quantile(rhos, 0.975))

    return {"n": n, "rho": float(base_rho), "ci_lo": ci_lo, "ci_hi": ci_hi}

def plot_lambda_curve(df_curve: pd.DataFrame, title: str, out_png: Path, out_pdf: Path):

    plt.figure()

    plt.plot(df_curve["lambda"], df_curve["spearman_exclude_anchor"], marker="o")

    plt.xlabel("fp_penalty_lambda")

    plt.ylabel("Spearman (exclude anchor)")

    plt.title(title)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(out_png, dpi=200)

    plt.savefig(out_pdf)

    plt.close()

def plot_fp_rate_bar(fp_rate: pd.DataFrame, title: str, out_png: Path, out_pdf: Path):

    fp_rate = fp_rate.sort_values("fp_rate", ascending=False).copy()

    plt.figure(figsize=(10, 3.5))

    plt.bar(fp_rate["cultivar"].astype(str), fp_rate["fp_rate"].astype(float))

    plt.xlabel("Cultivar")

    plt.ylabel("FP_rate among true==0 VOCs")

    plt.title(title)

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(out_png, dpi=200)

    plt.savefig(out_pdf)

    plt.close()

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--run_dirs", nargs="+", required=True)

    ap.add_argument("--anchor", required=True)

    ap.add_argument("--out_tag", required=True)

    ap.add_argument("--space", required=True)

    ap.add_argument("--weight_mode", required=True)

    ap.add_argument("--fp_penalty_lambda", type=float, default=0.2)

    ap.add_argument("--lambda_grid", default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.4")

    ap.add_argument("--out_dir", default="results/paper_figs")

    ap.add_argument("--out_table_dir", default="results/paper_tables")

    ap.add_argument("--bootstrap_n", type=int, default=2000)

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    out_fig_dir = Path(args.out_dir)

    out_tbl_dir = Path(args.out_table_dir)

    out_fig_dir.mkdir(parents=True, exist_ok=True)

    out_tbl_dir.mkdir(parents=True, exist_ok=True)

    lambdas = [float(x) for x in args.lambda_grid.split(",") if x.strip() != ""]

    rows_ci = []

    rows_manifest = []

    for rd in args.run_dirs:

        run_dir = Path(rd)

        run_name = run_dir.name

        rank = load_rank_csv(

            run_dir=run_dir,

            out_tag=args.out_tag,

            space=args.space,

            weight_mode=args.weight_mode,

            fp_penalty_lambda=args.fp_penalty_lambda,

        )

        pv, pv_path = load_pv_thresh(run_dir)

        fp_rate_df = compute_fp_rate(pv)

        curve = penalty_curve(rank, fp_rate_df, args.anchor, lambdas, base_lambda=args.fp_penalty_lambda)

        curve_path = out_tbl_dir / f"robust_lambda_curve.{run_name}.{args.space}.{args.weight_mode}.csv"

        curve.to_csv(curve_path, index=False)

        plot_lambda_curve(

            curve,

            title=f"{run_name} | {args.space} {args.weight_mode}",

            out_png=out_fig_dir / f"fig_lambda_curve.{run_name}.{args.space}.{args.weight_mode}.png",

            out_pdf=out_fig_dir / f"fig_lambda_curve.{run_name}.{args.space}.{args.weight_mode}.pdf",

        )

        plot_fp_rate_bar(

            fp_rate_df,

            title=f"{run_name} FP-rate (hallucination proxy)",

            out_png=out_fig_dir / f"fig_fp_rate.{run_name}.png",

            out_pdf=out_fig_dir / f"fig_fp_rate.{run_name}.pdf",

        )

        score_col = "pred_sim_pen" if "pred_sim_pen" in rank.columns else "pred_sim"

        ci = bootstrap_ci_spearman(rank, anchor=args.anchor, score_col=score_col, n_boot=args.bootstrap_n, seed=args.seed)

        ci.update({

            "run_name": run_name,

            "score_col": score_col,

            "space": args.space,

            "weight_mode": args.weight_mode,

            "fp_penalty_lambda": args.fp_penalty_lambda,

        })

        rows_ci.append(ci)

        rows_manifest.append({

            "run_name": run_name,

            "run_dir": str(run_dir),

            "ranking_csv": str((run_dir / "ideotype_v3" / f"ideotype_ranking_v3.{args.out_tag}.fpPen{args.fp_penalty_lambda:0.2f}.{args.space}.{args.weight_mode}.csv").resolve()),

            "pv_thresh_parquet": str(pv_path.resolve()),

            "score_col_used": score_col,

        })

        print(f"[OK] robustness for {run_name}")

        print(f"  - lambda curve: {curve_path}")

        print(f"  - figs saved  : fig_lambda_curve / fig_fp_rate in {out_fig_dir}")

    df_ci = pd.DataFrame(rows_ci)

    out_ci = out_tbl_dir / f"robust_bootstrap_ci.{args.space}.{args.weight_mode}.fpPen{args.fp_penalty_lambda:0.2f}.csv"

    df_ci.to_csv(out_ci, index=False)

    df_manifest = pd.DataFrame(rows_manifest)

    out_mani = out_tbl_dir / f"robust_manifest.{args.space}.{args.weight_mode}.fpPen{args.fp_penalty_lambda:0.2f}.csv"

    df_manifest.to_csv(out_mani, index=False)

    print("\n[OK] wrote:")

    print(f"  - {out_ci}")

    print(f"  - {out_mani}")

    print(f"[DONE] robustness figs in: {out_fig_dir}")

if __name__ == "__main__":

    main()
