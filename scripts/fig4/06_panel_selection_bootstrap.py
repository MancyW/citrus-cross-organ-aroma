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

def load_PT(pred_long_path: Path):

    df = pd.read_parquet(pred_long_path)

    W_pred = df.pivot_table(index="cultivar", columns="VOC", values="y_pred", aggfunc="first").fillna(0.0).sort_index()

    W_true = df.pivot_table(index="cultivar", columns="VOC", values="y_true", aggfunc="first").fillna(0.0).reindex(W_pred.index)

    P = np.log1p(W_pred.to_numpy(float))

    T = np.log1p(W_true.to_numpy(float))

    cultivars = W_true.index.to_numpy()

    return cultivars, P, T, W_pred.to_numpy(float), W_true.to_numpy(float)

def eval_panel(pred_long_path: Path) -> dict:

    cultivars, P, T, Wp, Wt = load_PT(pred_long_path)

    n = T.shape[0]

    cos_list = np.array([cosine(P[i], T[i]) for i in range(n)], float)

    r_list   = np.array([pearson(P[i], T[i]) for i in range(n)], float)

    sumT = T.sum(axis=0, keepdims=True)

    mu_train = (sumT - T) / float(n - 1)

    base_cos_list = np.array([cosine(mu_train[i], T[i]) for i in range(n)], float)

    base_r_list   = np.array([pearson(mu_train[i], T[i]) for i in range(n)], float)

    vc = vocwise_corr(P, T)

    vc_med = float(np.median(vc)) if vc.size else float("nan")

    vc_frac_neg = float((vc < 0).mean()) if vc.size else float("nan")

    pred_frac_pos = float((Wp > 0).mean())

    true_frac_pos = float((Wt > 0).mean())

    sparsity_ratio = float(pred_frac_pos / (true_frac_pos + 1e-12))

    return {

        "n_cultivars": int(n),

        "n_vocs": int(T.shape[1]),

        "mean_cosine": float(cos_list.mean()),

        "std_cosine": float(cos_list.std()),

        "mean_pearson": float(r_list.mean()),

        "std_pearson": float(r_list.std()),

        "baseline_loco_mean_cosine": float(base_cos_list.mean()),

        "baseline_loco_mean_pearson": float(base_r_list.mean()),

        "delta_cosine_vs_loco_baseline": float(cos_list.mean() - base_cos_list.mean()),

        "delta_pearson_vs_loco_baseline": float(r_list.mean() - base_r_list.mean()),

        "voc_corr_n": int(vc.size),

        "voc_corr_median": vc_med,

        "voc_corr_frac_neg": vc_frac_neg,

        "pred_frac_pos": pred_frac_pos,

        "true_frac_pos": true_frac_pos,

        "sparsity_ratio": sparsity_ratio,

    }

def bootstrap_panel_meancos(P: np.ndarray, T: np.ndarray, B: int, seed: int):

    rng = np.random.default_rng(seed)

    n = T.shape[0]

    stats = np.empty(B, float)

    for b in range(B):

        idx = rng.integers(0, n, size=n)

        stats[b] = np.mean([cosine(P[i], T[i]) for i in idx])

    return stats

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--root", required=True)

    ap.add_argument("--B", type=int, default=5000)

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    root = Path(args.root)

    if not root.exists():

        raise SystemExit(f"root not found: {root}")

    items = []

    for d in sorted([p for p in root.iterdir() if p.is_dir()]):

        pred_long = d / "pred_vectors_long.parquet"

        if pred_long.exists():

            items.append((panel_name_from_dir(d), d, pred_long))

    if not items:

        raise SystemExit(f"no panel outputs found under: {root}")

    rows = []

    cache_PT = {}

    for panel, d, pred_long in items:

        m = eval_panel(pred_long)

        m["panel"] = panel

        m["dir"] = str(d)

        rows.append(m)

        cultivars, P, T, _, _ = load_PT(pred_long)

        cache_PT[panel] = (P, T)

    rep = pd.DataFrame(rows).sort_values(["mean_cosine", "mean_pearson"], ascending=False).reset_index(drop=True)

    panels = rep["panel"].tolist()

    boot = {}

    for panel in panels:

        P, T = cache_PT[panel]

        boot[panel] = bootstrap_panel_meancos(P, T, B=args.B, seed=args.seed)

    ci_lo = []

    ci_hi = []

    for panel in panels:

        x = boot[panel]

        ci_lo.append(float(np.percentile(x, 2.5)))

        ci_hi.append(float(np.percentile(x, 97.5)))

    rep["mean_cosine_ci95_lo"] = ci_lo

    rep["mean_cosine_ci95_hi"] = ci_hi

    B = args.B

    mat = np.vstack([boot[p] for p in panels])

    best_idx = np.argmax(mat, axis=0)

    best_prob = [(best_idx == i).mean() for i in range(len(panels))]

    rep["prob_best_by_mean_cosine"] = best_prob

    top1 = panels[0]

    top2 = panels[1] if len(panels) > 1 else None

    if top2 is not None:

        diff = boot[top1] - boot[top2]

        rep.attrs["top1"] = top1

        rep.attrs["top2"] = top2

        rep.attrs["diff_top1_minus_top2_mean"] = float(diff.mean())

        rep.attrs["diff_top1_minus_top2_ci95"] = (

            float(np.percentile(diff, 2.5)),

            float(np.percentile(diff, 97.5)),

        )

        rep.attrs["p_top2_beats_top1"] = float((diff < 0).mean())

    out_csv = root / "panel_ablation_report_with_ci.csv"

    rep.to_csv(out_csv, index=False)

    print("=== Saved ===")

    print(out_csv)

    show_cols = [

        "panel","mean_cosine","mean_cosine_ci95_lo","mean_cosine_ci95_hi",

        "prob_best_by_mean_cosine",

        "mean_pearson",

        "baseline_loco_mean_cosine","delta_cosine_vs_loco_baseline",

        "sparsity_ratio","voc_corr_median","voc_corr_frac_neg"

    ]

    print("\n=== Top 15 (with CI) ===")

    print(rep[show_cols].head(15).to_string(index=False))

    if top2 is not None:

        lo, hi = rep.attrs["diff_top1_minus_top2_ci95"]

        print("\n=== Top1 vs Top2 bootstrap ===")

        print(f"top1={top1}  top2={top2}")

        print(f"diff(mean_cosine) = {rep.attrs['diff_top1_minus_top2_mean']:.6f}")

        print(f"95% CI            = [{lo:.6f}, {hi:.6f}]")

        print(f"P(top2 > top1)    = {rep.attrs['p_top2_beats_top1']:.4f}")

if __name__ == "__main__":

    main()
