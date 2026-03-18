from __future__ import annotations

import argparse

from pathlib import Path

import json

import numpy as np

import pandas as pd

from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score

def onehot(df: pd.DataFrame, cols: list[str]) -> np.ndarray:

    return pd.get_dummies(df[cols].astype(str), drop_first=False).to_numpy(dtype=float)

def fit_r2(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> float:

    m = Ridge(alpha=alpha)

    m.fit(X, y)

    yhat = m.predict(X)

    return float(r2_score(y, yhat))

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--ssot_long", default="data/ssot/ssot_long.parquet")

    ap.add_argument("--outdir", default="results/leaf_stage_effect")

    ap.add_argument("--alpha", type=float, default=1.0)

    ap.add_argument("--log1p", action="store_true")

    args = ap.parse_args()

    outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.ssot_long)

    df_leaf = df[df["Organ"] == "Leaf"].copy()

    meta = {"SampleID", "PairID", "Cultivar", "Organ", "Stage", "Batch", "Rep"}

    voc_cols = [c for c in df_leaf.columns if c not in meta]

    voc_cols = sorted(voc_cols)

    X_stage = onehot(df_leaf, ["Stage"])

    X_cult = onehot(df_leaf, ["Cultivar"])

    X_both = onehot(df_leaf, ["Cultivar", "Stage"])

    rows = []

    for v in voc_cols:

        y = df_leaf[v].to_numpy(dtype=float)

        if args.log1p:

            y = np.log1p(y)

        r2_stage = fit_r2(X_stage, y, alpha=args.alpha)

        r2_cult = fit_r2(X_cult, y, alpha=args.alpha)

        r2_both = fit_r2(X_both, y, alpha=args.alpha)

        rows.append({

            "VOC": v,

            "R2_stage": r2_stage,

            "R2_cultivar": r2_cult,

            "R2_both": r2_both,

            "delta_stage_given_cultivar": max(0.0, r2_both - r2_cult),

            "delta_cultivar_given_stage": max(0.0, r2_both - r2_stage),

        })

    out = pd.DataFrame(rows).sort_values("delta_stage_given_cultivar", ascending=False)

    out.to_csv(outdir / "leaf_variance_partition.csv", index=False)

    summary = {

        "n_leaf_samples": int(df_leaf.shape[0]),

        "n_vocs": int(len(voc_cols)),

        "alpha": args.alpha,

        "log1p": bool(args.log1p),

        "median_delta_stage_given_cultivar": float(out["delta_stage_given_cultivar"].median()),

        "median_delta_cultivar_given_stage": float(out["delta_cultivar_given_stage"].median()),

        "top10_stage_driven_vocs": out.head(10)[["VOC","delta_stage_given_cultivar"]].to_dict(orient="records"),

    }

    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Saved:")

    print("  -", outdir / "leaf_variance_partition.csv")

    print("  -", outdir / "summary.json")

if __name__ == "__main__":

    main()
