from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

def pearson(x: np.ndarray, y: np.ndarray) -> float:

    x = np.asarray(x, dtype=float)

    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)

    x = x[m]; y = y[m]

    if len(x) < 3:

        return np.nan

    x = x - x.mean()

    y = y - y.mean()

    denom = np.sqrt((x*x).sum()) * np.sqrt((y*y).sum())

    if denom == 0:

        return np.nan

    return float((x*y).sum() / denom)

def spearman(x: np.ndarray, y: np.ndarray) -> float:

    rx = pd.Series(x).rank(method="average").to_numpy()

    ry = pd.Series(y).rank(method="average").to_numpy()

    return pearson(rx, ry)

def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument("--outdir", required=True, type=str)

    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()

    df = pd.read_csv(outdir / "failure" / "cultivar_error_with_doa.csv")

    req = ["Cultivar", "abs_error_peel_index", "vector_mae_log1p", "doa_knn_distance"]

    for c in req:

        if c not in df.columns:

            raise ValueError(f"Missing column {c} in cultivar_error_with_doa.csv")

    x = df["doa_knn_distance"].astype(float).to_numpy()

    y1 = df["abs_error_peel_index"].astype(float).to_numpy()

    y2 = df["vector_mae_log1p"].astype(float).to_numpy()

    stats = []

    stats.append({"target": "abs_error_peel_index", "pearson": pearson(x, y1), "spearman": spearman(x, y1), "n": int(np.isfinite(x).sum())})

    stats.append({"target": "vector_mae_log1p", "pearson": pearson(x, y2), "spearman": spearman(x, y2), "n": int(np.isfinite(x).sum())})

    def r2(xv, yv):

        m = np.isfinite(xv) & np.isfinite(yv)

        xv = xv[m]; yv = yv[m]

        if len(xv) < 3:

            return np.nan

        A = np.vstack([np.ones_like(xv), xv]).T

        beta, *_ = np.linalg.lstsq(A, yv, rcond=None)

        yhat = A @ beta

        ss_res = float(((yv - yhat) ** 2).sum())

        ss_tot = float(((yv - yv.mean()) ** 2).sum())

        return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    stats[0]["r2_linear"] = r2(x, y1)

    stats[1]["r2_linear"] = r2(x, y2)

    out_csv = pd.DataFrame(stats)

    out_path = outdir / "failure" / "doa_error_model.csv"

    out_csv.to_csv(out_path, index=False)

    try:

        import matplotlib.pyplot as plt

        plt.figure()

        plt.scatter(x, y1, alpha=0.8)

        plt.xlabel("DOA kNN distance")

        plt.ylabel("abs error (peel index)")

        plt.title(f"DOA vs error (Spearman={stats[0]['spearman']:.3f}, R2={stats[0]['r2_linear']:.3f})")

        plt.tight_layout()

        fig_out = outdir / "failure" / "doa_error_scatter.png"

        plt.savefig(fig_out, dpi=220)

        plt.close()

    except Exception:

        pass

    print(f"[OK] wrote: {out_path}")

if __name__ == "__main__":

    main()
