from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

from numpy.linalg import norm

def cosine(a, b, eps=1e-12):

    return float(np.dot(a, b) / ((norm(a) + eps) * (norm(b) + eps)))

def pearson(a, b, eps=1e-12):

    a = a - a.mean()

    b = b - b.mean()

    return float(np.dot(a, b) / ((norm(a) + eps) * (norm(b) + eps)))

def load_PT(pred_long_path: Path):

    df = pd.read_parquet(pred_long_path)

    W_pred = df.pivot_table(index="cultivar", columns="VOC", values="y_pred", aggfunc="first").fillna(0.0).sort_index()

    W_true = df.pivot_table(index="cultivar", columns="VOC", values="y_true", aggfunc="first").fillna(0.0).reindex(W_pred.index)

    P = np.log1p(W_pred.to_numpy(float))

    T = np.log1p(W_true.to_numpy(float))

    return P, T

def panel_name_from_dir(d: Path) -> str:

    return d.name.replace("__", "+")

def eval_skill(P: np.ndarray, T: np.ndarray) -> dict:

    n = T.shape[0]

    sumT = T.sum(axis=0, keepdims=True)

    mu_train = (sumT - T) / float(n - 1)

    r_true = T - mu_train

    r_pred = P - mu_train

    cos_res = np.array([cosine(r_pred[i], r_true[i]) for i in range(n)], float)

    r_res   = np.array([pearson(r_pred[i], r_true[i]) for i in range(n)], float)

    cos_raw = np.array([cosine(P[i], T[i]) for i in range(n)], float)

    r_raw   = np.array([pearson(P[i], T[i]) for i in range(n)], float)

    return {

        "mean_cosine_raw": float(cos_raw.mean()),

        "mean_pearson_raw": float(r_raw.mean()),

        "mean_cosine_residual": float(cos_res.mean()),

        "mean_pearson_residual": float(r_res.mean()),

        "std_cosine_residual": float(cos_res.std()),

    }

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--root", required=True, help="e.g. results/ablation_allpanels/loco_YYYYMMDD_HHMMSS")

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

        P, T = load_PT(pred_long)

        m = eval_skill(P, T)

        m["panel"] = panel

        m["dir"] = str(d)

        rows.append(m)

    rep = pd.DataFrame(rows).sort_values("mean_cosine_residual", ascending=False).reset_index(drop=True)

    out = root / "panel_skill_over_baseline.csv"

    rep.to_csv(out, index=False)

    print("=== Saved ===")

    print(out)

    print("\n=== Top panels by RESIDUAL skill (higher = better cultivar-specific signal) ===")

    print(rep[[

        "panel",

        "mean_cosine_residual", "mean_pearson_residual", "std_cosine_residual",

        "mean_cosine_raw", "mean_pearson_raw"

    ]].head(15).to_string(index=False))

if __name__ == "__main__":

    main()
