from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

def _read_table(p: Path) -> pd.DataFrame:

    if p.suffix.lower() == ".parquet":

        return pd.read_parquet(p)

    return pd.read_csv(p)

def _write_table(df: pd.DataFrame, out: Path) -> None:

    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".parquet":

        df.to_parquet(out, index=False)

    else:

        df.to_csv(out, index=False)

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--pred_long", required=True, help="pred_vectors_long.parquet or .csv")

    ap.add_argument("--out", default=None, help="output path (default: alongside input with suffix)")

    ap.add_argument("--quantile", type=float, default=0.05,

                    help="VOC-specific threshold = q-quantile of TRUE positive values (default 0.05)")

    ap.add_argument("--space", default="absolute", choices=["absolute", "log1p"],

                    help="thresholding space: absolute compares y_pred; log1p compares log1p(y_pred)")

    ap.add_argument("--min_pos", type=int, default=1,

                    help="minimum number of TRUE positive samples required to define a threshold; "

                         "if fewer than min_pos, threshold is set to +inf (VOC predicted all zero). "

                         "Default 1 means use any available positive.")

    ap.add_argument("--keep_raw", action="store_true",

                    help="keep original y_pred in column y_pred_raw and overwrite y_pred with thresholded")

    ap.add_argument("--report", action="store_true",

                    help="print before/after sparsity and quick VOC-wise corr stats")

    args = ap.parse_args()

    p = Path(args.pred_long)

    df = _read_table(p)

    need = {"cultivar", "VOC", "y_pred", "y_true"}

    miss = need - set(df.columns)

    if miss:

        raise KeyError(f"Missing columns in {p}: {sorted(miss)}")

    q = float(args.quantile)

    if not (0.0 <= q <= 1.0):

        raise ValueError("--quantile must be in [0,1]")

    vocs = df["VOC"].astype(str).unique().tolist()

    thr = {}

    npos = {}

    if args.space == "absolute":

        true_vals = df["y_true"].to_numpy(dtype=float)

        pred_vals = df["y_pred"].to_numpy(dtype=float)

        for v in vocs:

            tpos = df.loc[(df["VOC"] == v) & (df["y_true"] > 0), "y_true"].to_numpy(dtype=float)

            npos[v] = int(tpos.size)

            if tpos.size < int(args.min_pos):

                thr[v] = float("inf")

            else:

                thr[v] = float(np.quantile(tpos, q))

    else:

        for v in vocs:

            tpos = df.loc[(df["VOC"] == v) & (df["y_true"] > 0), "y_true"].to_numpy(dtype=float)

            npos[v] = int(tpos.size)

            if tpos.size < int(args.min_pos):

                thr[v] = float("inf")

            else:

                thr[v] = float(np.quantile(np.log1p(tpos), q))

    if args.keep_raw:

        if "y_pred_raw" not in df.columns:

            df.insert(df.columns.get_loc("y_pred") + 1, "y_pred_raw", df["y_pred"].to_numpy(dtype=float))

        else:

            df["y_pred_raw"] = df["y_pred"].to_numpy(dtype=float)

    y_pred_new = df["y_pred"].to_numpy(dtype=float).copy()

    if args.space == "absolute":

        for i, (v, yp) in enumerate(zip(df["VOC"].astype(str).to_list(), y_pred_new)):

            t = thr.get(v, float("inf"))

            if not np.isfinite(t):

                y_pred_new[i] = 0.0

            else:

                if yp < t:

                    y_pred_new[i] = 0.0

    else:

        yp_log = np.log1p(np.clip(y_pred_new, 0.0, None))

        for i, v in enumerate(df["VOC"].astype(str).to_list()):

            t = thr.get(v, float("inf"))

            if not np.isfinite(t):

                y_pred_new[i] = 0.0

            else:

                if yp_log[i] < t:

                    y_pred_new[i] = 0.0

    df["y_pred"] = y_pred_new

    if args.out:

        out = Path(args.out)

    else:

        suffix = f".thresh_q{q:g}_{args.space}.parquet" if p.suffix.lower() == ".parquet" else f".thresh_q{q:g}_{args.space}.csv"

        out = p.with_name(p.stem + suffix)

    _write_table(df, out)

    if args.report:

        Wp = df.pivot_table(index="cultivar", columns="VOC", values="y_pred", aggfunc="first").fillna(0.0)

        Wt = df.pivot_table(index="cultivar", columns="VOC", values="y_true", aggfunc="first").fillna(0.0).reindex(Wp.index).reindex(columns=Wp.columns)

        pred_pos = float((Wp.to_numpy(float) > 0).mean())

        true_pos = float((Wt.to_numpy(float) > 0).mean())

        P = np.log1p(Wp.to_numpy(float))

        T = np.log1p(Wt.to_numpy(float))

        cors = []

        for j in range(P.shape[1]):

            a, b = P[:, j], T[:, j]

            if np.std(a) < 1e-12 or np.std(b) < 1e-12:

                continue

            cors.append(float(np.corrcoef(a, b)[0, 1]))

        cors = np.array(cors, dtype=float)

        print("[REPORT]")

        print("  output:", out)

        print("  pred fraction>0:", pred_pos)

        print("  true fraction>0:", true_pos)

        print("  ratio(pred/true):", pred_pos / (true_pos + 1e-12))

        print("  VOC-wise corr(log1p) n =", int(cors.size),

              "median =", float(np.median(cors)) if cors.size else float("nan"),

              "frac<0 =", float((cors < 0).mean()) if cors.size else float("nan"))

        n_inf = sum(1 for v in vocs if not np.isfinite(thr.get(v, float("inf"))))

        print("  thresholds: nVOC =", len(vocs),

              "| nVOC(threshold=inf) =", n_inf,

              "| quantile =", q,

              "| space =", args.space,

              "| min_pos =", int(args.min_pos))

    print("[OK] Saved:", out)

if __name__ == "__main__":

    main()
