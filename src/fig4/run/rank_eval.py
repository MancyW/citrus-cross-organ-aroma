from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

def spearman(x: np.ndarray, y: np.ndarray) -> float:

    rx = pd.Series(x).rank(method="average").to_numpy()

    ry = pd.Series(y).rank(method="average").to_numpy()

    rx = rx - rx.mean()

    ry = ry - ry.mean()

    denom = (np.linalg.norm(rx) * np.linalg.norm(ry)) + 1e-12

    return float(np.dot(rx, ry) / denom)

def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:

    n = len(x)

    concord = 0

    discord = 0

    for i in range(n):

        for j in range(i + 1, n):

            a = x[i] - x[j]

            b = y[i] - y[j]

            s = a * b

            if s > 0:

                concord += 1

            elif s < 0:

                discord += 1

    denom = concord + discord

    if denom == 0:

        return 0.0

    return float((concord - discord) / denom)

def topk_metrics(true_score: np.ndarray, pred_score: np.ndarray, k: int) -> dict:

    k = int(k)

    k = max(1, min(k, len(true_score)))

    top_pred = np.argsort(-pred_score)[:k]

    top_true = np.argsort(-true_score)[:k]

    recall = len(set(top_pred).intersection(set(top_true))) / float(k)

    enrich = (true_score[top_pred].mean() + 1e-12) / (true_score.mean() + 1e-12)

    return {"k": k, "recall": float(recall), "enrichment": float(enrich)}

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--pred_csv", required=True, help="results/runs/<run_id>/predictions_summary.csv")

    ap.add_argument("--outdir", default=None)

    ap.add_argument("--ks", default="1,3,5")

    args = ap.parse_args()

    pred_path = Path(args.pred_csv)

    outdir = Path(args.outdir) if args.outdir else pred_path.parent / "rank_eval"

    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)

    need = {"panel", "cultivar", "true_score", "pred_score"}

    miss = need - set(df.columns)

    if miss:

        raise KeyError(f"Missing columns in {pred_path}: {sorted(miss)}")

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]

    rows = []

    chosen_rows = []

    for panel, g in df.groupby("panel"):

        true_score = g["true_score"].to_numpy(dtype=float)

        pred_score = g["pred_score"].to_numpy(dtype=float)

        sp = spearman(pred_score, true_score)

        kt = kendall_tau(pred_score, true_score)

        r = {

            "panel": panel,

            "n": int(len(g)),

            "spearman(pred,true)": float(sp),

            "kendall_tau(pred,true)": float(kt),

            "mean_cosine": float(g["cosine"].mean()) if "cosine" in g.columns else np.nan,

            "mean_pearson": float(g["pearson"].mean()) if "pearson" in g.columns else np.nan,

        }

        for k in ks:

            m = topk_metrics(true_score, pred_score, k=k)

            r[f"top{k}_recall"] = m["recall"]

            r[f"top{k}_enrichment"] = m["enrichment"]

            top_pred_idx = np.argsort(-pred_score)[:m["k"]]

            top_pred_cultivars = g.iloc[top_pred_idx]["cultivar"].tolist()

            chosen_rows.append({

                "panel": panel,

                "k": m["k"],

                "top_pred_cultivars": ";".join(top_pred_cultivars),

                "top_pred_mean_true_score": float(true_score[top_pred_idx].mean()),

                "overall_mean_true_score": float(true_score.mean()),

            })

        rows.append(r)

    out = pd.DataFrame(rows).sort_values(["mean_cosine", "spearman(pred,true)"], ascending=False)

    out.to_csv(outdir / "rank_eval.csv", index=False)

    chosen = pd.DataFrame(chosen_rows)

    chosen.to_csv(outdir / "topk_selected.csv", index=False)

    print("[OK] Saved:")

    print("  -", outdir / "rank_eval.csv")

    print("  -", outdir / "topk_selected.csv")

    print("\n[Preview] rank_eval (top 10):")

    print(out.head(10).to_string(index=False))

if __name__ == "__main__":

    main()
