from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

from src.fig4.cv.loco import iter_loco

def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:

    na = np.linalg.norm(a) + eps

    nb = np.linalg.norm(b) + eps

    return float(np.dot(a, b) / (na * nb))

def pearson(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:

    a = a - a.mean()

    b = b - b.mean()

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps

    return float(np.dot(a, b) / denom)

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--pred_long", required=True, help="pred_vectors_long.parquet OR .csv (for VOC list + true vectors)")

    ap.add_argument("--out", default=None)

    ap.add_argument("--log1p", action="store_true", help="evaluate cosine/pearson in log1p space")

    args = ap.parse_args()

    p = Path(args.pred_long)

    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)

    true_w = df.pivot_table(index="cultivar", columns="VOC", values="y_true", aggfunc="first").fillna(0.0).sort_index()

    cultivars = true_w.index.to_numpy()

    Y = true_w.to_numpy(dtype=float)

    groups = cultivars.copy()

    cos_list, r_list = [], []

    for tr_idx, te_idx in iter_loco(groups):

        Ytr = Y[tr_idx]

        ybar = Ytr.mean(axis=0)

        yt = Y[te_idx[0]]

        if args.log1p:

            yt = np.log1p(yt)

            ybar_eval = np.log1p(ybar)

        else:

            ybar_eval = ybar

        cos_list.append(cosine(ybar_eval, yt))

        r_list.append(pearson(ybar_eval, yt))

    res = {

        "baseline": "mean_train_profile",

        "n": int(len(cultivars)),

        "mean_cosine": float(np.mean(cos_list)),

        "std_cosine": float(np.std(cos_list)),

        "mean_pearson": float(np.mean(r_list)),

        "std_pearson": float(np.std(r_list)),

        "eval_log1p": bool(args.log1p),

    }

    out_path = Path(args.out) if args.out else p.parent / "baseline_mean_loco.json"

    out_path.write_text(pd.Series(res).to_json(force_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Mean-baseline LOCO saved:", out_path)

    print(res)

if __name__ == "__main__":

    main()
