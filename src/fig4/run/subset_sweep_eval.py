from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:

    na = np.linalg.norm(a) + eps

    nb = np.linalg.norm(b) + eps

    return float(np.dot(a, b) / (na * nb))

def pearson(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:

    a = a - a.mean()

    b = b - b.mean()

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps

    return float(np.dot(a, b) / denom)

def spearman(x: np.ndarray, y: np.ndarray) -> float:

    rx = pd.Series(x).rank(method="average").to_numpy()

    ry = pd.Series(y).rank(method="average").to_numpy()

    rx = rx - rx.mean()

    ry = ry - ry.mean()

    denom = (np.linalg.norm(rx) * np.linalg.norm(ry)) + 1e-12

    return float(np.dot(rx, ry) / denom)

def topk_recall(true: np.ndarray, pred: np.ndarray, k: int) -> float:

    k = max(1, min(int(k), len(true)))

    top_pred = np.argsort(-pred)[:k]

    top_true = np.argsort(-true)[:k]

    return len(set(top_pred).intersection(set(top_true))) / float(k)

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--pred_long", required=True, help="pred_vectors_long.parquet or .csv (contains y_true and y_pred)")

    ap.add_argument("--outdir", default=None)

    ap.add_argument("--Ns", default="10,20,30,50,80,120", help="subset sizes to test")

    ap.add_argument("--score_mode", default="weighted_sum", choices=["weighted_sum"])

    ap.add_argument("--weights", default="var", choices=["var", "cv"], help="how to weight VOCs in breeding score")

    ap.add_argument("--log1p_eval", action="store_true", help="evaluate cos/r in log1p space")

    args = ap.parse_args()

    p = Path(args.pred_long)

    outdir = Path(args.outdir) if args.outdir else p.parent / "subset_sweep"

    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)

    true_w = df.pivot_table(index="cultivar", columns="VOC", values="y_true", aggfunc="first").fillna(0.0).sort_index()

    pred_w = df.pivot_table(index="cultivar", columns="VOC", values="y_pred", aggfunc="first").fillna(0.0).reindex(true_w.index)

    cultivars = true_w.index.tolist()

    vocs = true_w.columns.tolist()

    Yt = true_w.to_numpy(dtype=float)

    Yp = pred_w.to_numpy(dtype=float)

    Ymean = np.repeat(Yt.mean(axis=0, keepdims=True), repeats=Yt.shape[0], axis=0)

    mu = Yt.mean(axis=0)

    var = Yt.var(axis=0)

    sd = Yt.std(axis=0)

    cv = sd / (mu + 1e-12)

    disc_df = pd.DataFrame({"VOC": vocs, "var": var, "cv": cv, "mean": mu})

    disc_df = disc_df.sort_values(args.weights, ascending=False).reset_index(drop=True)

    disc_df.to_csv(outdir / "voc_discriminativeness.csv", index=False)

    Ns = [int(x.strip()) for x in args.Ns.split(",") if x.strip()]

    rows = []

    for N in Ns:

        N = max(2, min(N, len(vocs)))

        subset = disc_df.head(N)["VOC"].tolist()

        idx = [vocs.index(v) for v in subset]

        yt = Yt[:, idx]

        yp = Yp[:, idx]

        yb = Ymean[:, idx]

        if args.log1p_eval:

            yt_eval = np.log1p(yt)

            yp_eval = np.log1p(yp)

            yb_eval = np.log1p(yb)

        else:

            yt_eval, yp_eval, yb_eval = yt, yp, yb

        cos_m = np.mean([cosine(yp_eval[i], yt_eval[i]) for i in range(len(cultivars))])

        r_m = np.mean([pearson(yp_eval[i], yt_eval[i]) for i in range(len(cultivars))])

        cos_b = np.mean([cosine(yb_eval[i], yt_eval[i]) for i in range(len(cultivars))])

        r_b = np.mean([pearson(yb_eval[i], yt_eval[i]) for i in range(len(cultivars))])

        w = disc_df.head(N)[args.weights].to_numpy(dtype=float)

        w = w / (w.sum() + 1e-12)

        true_score = (np.log1p(yt) if args.log1p_eval else yt) @ w

        pred_score = (np.log1p(yp) if args.log1p_eval else yp) @ w

        base_score = (np.log1p(yb) if args.log1p_eval else yb) @ w

        sp_model = spearman(pred_score, true_score)

        sp_base = spearman(base_score, true_score)

        rows.append({

            "N": N,

            "weights": args.weights,

            "log1p_eval": bool(args.log1p_eval),

            "model_mean_cos": float(cos_m),

            "baseline_mean_cos": float(cos_b),

            "delta_cos": float(cos_m - cos_b),

            "model_mean_r": float(r_m),

            "baseline_mean_r": float(r_b),

            "delta_r": float(r_m - r_b),

            "spearman_model": float(sp_model),

            "spearman_baseline": float(sp_base),

            "top1_recall_model": float(topk_recall(true_score, pred_score, 1)),

            "top3_recall_model": float(topk_recall(true_score, pred_score, 3)),

            "top5_recall_model": float(topk_recall(true_score, pred_score, 5)),

        })

    out = pd.DataFrame(rows).sort_values("N")

    out.to_csv(outdir / "subset_sweep_metrics.csv", index=False)

    print("[OK] Saved:")

    print("  -", outdir / "voc_discriminativeness.csv")

    print("  -", outdir / "subset_sweep_metrics.csv")

    print("\n[Preview] subset_sweep_metrics:")

    print(out.to_string(index=False))

if __name__ == "__main__":

    main()
