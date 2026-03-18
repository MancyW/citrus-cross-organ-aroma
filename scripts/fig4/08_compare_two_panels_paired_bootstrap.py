from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

from numpy.linalg import norm

def cosine(a, b, eps=1e-12) -> float:

    return float(np.dot(a, b) / ((norm(a) + eps) * (norm(b) + eps)))

def pearson(a, b, eps=1e-12) -> float:

    a = a - a.mean()

    b = b - b.mean()

    return float(np.dot(a, b) / ((norm(a) + eps) * (norm(b) + eps)))

def panel_to_dirname(panel: str) -> str:

    return panel.replace("+", "__")

def load_panel_arrays(root: Path, panel: str):

    d = root / panel_to_dirname(panel)

    p = d / "pred_vectors_long.parquet"

    if not p.exists():

        raise FileNotFoundError(f"pred_vectors_long.parquet not found for panel={panel} at {p}")

    df = pd.read_parquet(p)

    W_pred = (

        df.pivot_table(index="cultivar", columns="VOC", values="y_pred", aggfunc="first")

        .fillna(0.0)

        .sort_index()

    )

    W_true = (

        df.pivot_table(index="cultivar", columns="VOC", values="y_true", aggfunc="first")

        .fillna(0.0)

        .reindex(W_pred.index)

    )

    P = np.log1p(W_pred.to_numpy(float))

    T = np.log1p(W_true.to_numpy(float))

    cultivars = W_pred.index.to_numpy()

    return cultivars, P, T

def per_cultivar_metrics(P: np.ndarray, T: np.ndarray):

    n = T.shape[0]

    sumT = T.sum(axis=0, keepdims=True)

    mu_train = (sumT - T) / float(n - 1)

    raw_cos = np.array([cosine(P[i], T[i]) for i in range(n)], float)

    raw_r   = np.array([pearson(P[i], T[i]) for i in range(n)], float)

    r_true = T - mu_train

    r_pred = P - mu_train

    res_cos = np.array([cosine(r_pred[i], r_true[i]) for i in range(n)], float)

    res_r   = np.array([pearson(r_pred[i], r_true[i]) for i in range(n)], float)

    return raw_cos, raw_r, res_cos, res_r

def bootstrap_diff(a: np.ndarray, b: np.ndarray, B: int, seed: int):

    rng = np.random.default_rng(seed)

    n = a.shape[0]

    diffs = np.empty(B, float)

    for t in range(B):

        idx = rng.integers(0, n, size=n)

        diffs[t] = float(a[idx].mean() - b[idx].mean())

    lo, hi = np.percentile(diffs, [2.5, 97.5])

    return float(diffs.mean()), float(lo), float(hi), float((diffs > 0).mean())

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--root", required=True, help="e.g. results/ablation_allpanels/loco_YYYYMMDD_HHMMSS")

    ap.add_argument("--panel_a", required=True, help='e.g. "S4" or "S1+S4"')

    ap.add_argument("--panel_b", required=True, help='e.g. "S1+S4"')

    ap.add_argument("--B", type=int, default=10000)

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    root = Path(args.root)

    ca, Pa, Ta = load_panel_arrays(root, args.panel_a)

    cb, Pb, Tb = load_panel_arrays(root, args.panel_b)

    if ca.shape[0] != cb.shape[0] or not np.all(ca == cb):

        raise SystemExit("Cultivar index mismatch between two panels. Check pred_vectors_long files.")

    a_raw_cos, a_raw_r, a_res_cos, a_res_r = per_cultivar_metrics(Pa, Ta)

    b_raw_cos, b_raw_r, b_res_cos, b_res_r = per_cultivar_metrics(Pb, Tb)

    print("=== Panels ===")

    print("A =", args.panel_a)

    print("B =", args.panel_b)

    print("n_cultivars =", ca.shape[0])

    print("\n=== Point estimates (mean over cultivars) ===")

    print(f"raw_cos: A={a_raw_cos.mean():.6f}  B={b_raw_cos.mean():.6f}  diff(A-B)={(a_raw_cos.mean()-b_raw_cos.mean()):.6f}")

    print(f"raw_r  : A={a_raw_r.mean():.6f}    B={b_raw_r.mean():.6f}    diff(A-B)={(a_raw_r.mean()-b_raw_r.mean()):.6f}")

    print(f"res_cos: A={a_res_cos.mean():.6f}  B={b_res_cos.mean():.6f}  diff(A-B)={(a_res_cos.mean()-b_res_cos.mean()):.6f}")

    print(f"res_r  : A={a_res_r.mean():.6f}    B={b_res_r.mean():.6f}    diff(A-B)={(a_res_r.mean()-b_res_r.mean()):.6f}")

    print("\n=== Paired bootstrap on mean differences (A - B) ===")

    for name, aa, bb in [

        ("raw_cos", a_raw_cos, b_raw_cos),

        ("raw_r",   a_raw_r,   b_raw_r),

        ("res_cos", a_res_cos, b_res_cos),

        ("res_r",   a_res_r,   b_res_r),

    ]:

        m, lo, hi, p = bootstrap_diff(aa, bb, B=args.B, seed=args.seed + hash(name) % 100000)

        print(f"{name:7s}: mean_diff={m:.6f}  CI95=[{lo:.6f}, {hi:.6f}]  P(A>B)={p:.3f}")

    out = pd.DataFrame({

        "cultivar": ca,

        "A_raw_cos": a_raw_cos, "B_raw_cos": b_raw_cos,

        "A_raw_r": a_raw_r,     "B_raw_r": b_raw_r,

        "A_res_cos": a_res_cos, "B_res_cos": b_res_cos,

        "A_res_r": a_res_r,     "B_res_r": b_res_r,

    })

    out_path = root / f"paired_bootstrap_detail__{panel_to_dirname(args.panel_a)}__vs__{panel_to_dirname(args.panel_b)}.csv"

    out.to_csv(out_path, index=False)

    print("\n=== Saved per-cultivar detail ===")

    print(out_path)

if __name__ == "__main__":

    main()
