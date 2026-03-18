from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

def vocwise_corr_log1p(pred_w: pd.DataFrame, true_w: pd.DataFrame) -> np.ndarray:

    pred_w = pred_w.reindex(true_w.index).reindex(columns=true_w.columns)

    P = np.log1p(pred_w.to_numpy(float))

    T = np.log1p(true_w.to_numpy(float))

    cors = []

    for j in range(P.shape[1]):

        a, b = P[:, j], T[:, j]

        if np.std(a) < 1e-12 or np.std(b) < 1e-12:

            continue

        cors.append(np.corrcoef(a, b)[0, 1])

    return np.asarray(cors, dtype=float)

def reflect_global_log_space(pred_w: pd.DataFrame, true_w: pd.DataFrame) -> pd.DataFrame:

    true_w = true_w.sort_index()

    pred_w = pred_w.reindex(true_w.index).reindex(columns=true_w.columns)

    Tlog = np.log1p(true_w.to_numpy(float))

    Plog = np.log1p(pred_w.to_numpy(float))

    mu = Tlog.mean(axis=0, keepdims=True)

    Plog_fix = 2.0 * mu - Plog

    P_fix = np.expm1(Plog_fix)

    P_fix = np.clip(P_fix, 0.0, None)

    return pd.DataFrame(P_fix, index=true_w.index, columns=true_w.columns)

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--pred_long", required=True, help="pred_vectors_long.parquet OR .csv")

    ap.add_argument("--out", default=None, help="output parquet path")

    ap.add_argument("--keep_raw", action="store_true", help="keep y_pred_raw column")

    args = ap.parse_args()

    p = Path(args.pred_long)

    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)

    need = {"cultivar", "VOC", "y_pred", "y_true"}

    miss = need - set(df.columns)

    if miss:

        raise KeyError(f"Missing columns in {p}: {sorted(miss)}")

    if "panel" in df.columns:

        groups = list(df.groupby("panel", sort=False))

    else:

        groups = [(None, df)]

    out_parts = []

    for panel_val, g in groups:

        true_w = (

            g.pivot_table(index="cultivar", columns="VOC", values="y_true", aggfunc="first")

            .fillna(0.0)

            .sort_index()

        )

        pred_w = (

            g.pivot_table(index="cultivar", columns="VOC", values="y_pred", aggfunc="first")

            .fillna(0.0)

            .reindex(true_w.index)

            .reindex(columns=true_w.columns)

        )

        cors_before = vocwise_corr_log1p(pred_w, true_w)

        pred_fix_w = reflect_global_log_space(pred_w, true_w)

        cors_after = vocwise_corr_log1p(pred_fix_w, true_w)

        print(f"[PANEL={panel_val}] VOC-wise corr (log1p) before: n={cors_before.size} median={float(np.median(cors_before)):.4f}")

        print(f"[PANEL={panel_val}] VOC-wise corr (log1p) after : n={cors_after.size} median={float(np.median(cors_after)):.4f}")

        pred_fix_w.index.name = "cultivar"

        fix_long = pred_fix_w.reset_index().melt(

            id_vars=["cultivar"], var_name="VOC", value_name="y_pred_fix"

        )

        merged = g.merge(fix_long, on=["cultivar", "VOC"], how="left")

        if args.keep_raw:

            merged = merged.rename(columns={"y_pred": "y_pred_raw"})

            merged["y_pred"] = merged["y_pred_fix"]

        else:

            merged["y_pred"] = merged["y_pred_fix"]

        merged = merged.drop(columns=["y_pred_fix"])

        out_parts.append(merged)

    out_df = pd.concat(out_parts, ignore_index=True)

    out_path = Path(args.out) if args.out else p.with_name(p.stem + "_reflect_global_log.parquet")

    out_df.to_parquet(out_path, index=False)

    print("[OK] Saved:", out_path)

    print("  rows:", out_df.shape[0], "cols:", out_df.shape[1])

if __name__ == "__main__":

    main()
