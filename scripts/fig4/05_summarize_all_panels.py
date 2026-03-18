from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

from numpy.linalg import norm

def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:

    return float(np.dot(a, b) / ((norm(a) + eps) * (norm(b) + eps)))

def pearson(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:

    a = a - a.mean()

    b = b - b.mean()

    return float(np.dot(a, b) / ((norm(a) + eps) * (norm(b) + eps)))

def vocwise_corr(P: np.ndarray, T: np.ndarray) -> np.ndarray:

    cors = []

    for j in range(P.shape[1]):

        a, b = P[:, j], T[:, j]

        if np.std(a) < 1e-12 or np.std(b) < 1e-12:

            continue

        cors.append(np.corrcoef(a, b)[0, 1])

    return np.array(cors, dtype=float)

def panel_name_from_dir(d: Path) -> str:

    return d.name.replace("__", "+")

def eval_one(pred_long_path: Path) -> dict:

    df = pd.read_parquet(pred_long_path)

    W_pred = df.pivot_table(index="cultivar", columns="VOC", values="y_pred", aggfunc="first").fillna(0.0).sort_index()

    W_true = df.pivot_table(index="cultivar", columns="VOC", values="y_true", aggfunc="first").fillna(0.0).reindex(W_pred.index)

    P = np.log1p(W_pred.to_numpy(float))

    T = np.log1p(W_true.to_numpy(float))

    cos_list = [cosine(P[i], T[i]) for i in range(T.shape[0])]

    r_list   = [pearson(P[i], T[i]) for i in range(T.shape[0])]

    vc = vocwise_corr(P, T)

    vc_med = float(np.median(vc)) if vc.size else float("nan")

    vc_frac_neg = float((vc < 0).mean()) if vc.size else float("nan")

    vc_n = int(vc.size)

    pred_frac_pos = float((W_pred.to_numpy(float) > 0).mean())

    true_frac_pos = float((W_true.to_numpy(float) > 0).mean())

    sparsity_ratio = float(pred_frac_pos / (true_frac_pos + 1e-12))

    mu = T.mean(axis=0, keepdims=True)

    base_cos = float(np.mean([cosine(mu[0], T[i]) for i in range(T.shape[0])]))

    base_r   = float(np.mean([pearson(mu[0], T[i]) for i in range(T.shape[0])]))

    return {

        "mean_cosine": float(np.mean(cos_list)),

        "std_cosine": float(np.std(cos_list)),

        "mean_pearson": float(np.mean(r_list)),

        "std_pearson": float(np.std(r_list)),

        "baseline_mean_cosine": base_cos,

        "baseline_mean_pearson": base_r,

        "delta_cosine_vs_baseline": float(np.mean(cos_list) - base_cos),

        "delta_pearson_vs_baseline": float(np.mean(r_list) - base_r),

        "voc_corr_n": vc_n,

        "voc_corr_median": vc_med,

        "voc_corr_frac_neg": vc_frac_neg,

        "pred_frac_pos": pred_frac_pos,

        "true_frac_pos": true_frac_pos,

        "sparsity_ratio": sparsity_ratio,

        "n_cultivars": int(W_true.shape[0]),

        "n_vocs": int(W_true.shape[1]),

    }

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--root", required=True, help="e.g. results/ablation_allpanels/loco_YYYYMMDD_HHMMSS")

    ap.add_argument("--out", default=None, help="output csv path (default: <root>/panel_ablation_report.csv)")

    args = ap.parse_args()

    root = Path(args.root)

    if not root.exists():

        raise SystemExit(f"root not found: {root}")

    rows = []

    for d in sorted([p for p in root.iterdir() if p.is_dir()]):

        pred_long = d / "pred_vectors_long.parquet"

        if not pred_long.exists():

            continue

        panel = panel_name_from_dir(d)

        m = eval_one(pred_long)

        m["panel"] = panel

        m["dir"] = str(d)

        rows.append(m)

    if not rows:

        raise SystemExit(f"no panel outputs found under: {root}")

    rep = pd.DataFrame(rows)

    rep_acc = rep.sort_values(["mean_cosine", "mean_pearson"], ascending=False).reset_index(drop=True)

    out_path = Path(args.out) if args.out else (root / "panel_ablation_report.csv")

    rep_acc.to_csv(out_path, index=False)

    print("=== Saved report ===")

    print(out_path)

    print("\n=== Top panels by accuracy (LOCO) ===")

    show_cols = [

        "panel", "mean_cosine", "mean_pearson",

        "delta_cosine_vs_baseline",

        "pred_frac_pos", "true_frac_pos", "sparsity_ratio",

        "voc_corr_n", "voc_corr_median", "voc_corr_frac_neg",

    ]

    show_cols = [c for c in show_cols if c in rep_acc.columns]

    print(rep_acc[show_cols].head(15).to_string(index=False))

    best = rep_acc.iloc[0]

    print("\n[SELECT] Best-by-accuracy (LOCO):")

    print(f"  panel = {best['panel']}")

    print(f"  mean_cosine = {best['mean_cosine']:.6f}")

    print(f"  mean_pearson = {best['mean_pearson']:.6f}")

    print(f"  delta_cosine_vs_baseline = {best['delta_cosine_vs_baseline']:.6f}")

    print(f"  sparsity_ratio = {best['sparsity_ratio']:.3f}")

    print(f"  voc_corr_median = {best['voc_corr_median']:.3f} (n={int(best['voc_corr_n'])})")

if __name__ == "__main__":

    main()
