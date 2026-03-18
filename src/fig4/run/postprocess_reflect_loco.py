from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--pred_long", required=True, help="pred_vectors_long.parquet or .csv")

    ap.add_argument("--out", default=None, help="output file path (.parquet recommended)")

    ap.add_argument("--clip_nonneg", action="store_true", help="clip y_pred to >=0 after reflection")

    args = ap.parse_args()

    p = Path(args.pred_long)

    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)

    need = {"cultivar", "VOC", "y_true", "y_pred"}

    miss = need - set(df.columns)

    if miss:

        raise KeyError(f"Missing columns in {p}: {sorted(miss)}")

    has_panel = "panel" in df.columns

    panel_val = df["panel"].iloc[0] if has_panel else None

    true_w = (

        df.pivot_table(index="cultivar", columns="VOC", values="y_true", aggfunc="first")

        .fillna(0.0)

        .sort_index()

    )

    pred_w = (

        df.pivot_table(index="cultivar", columns="VOC", values="y_pred", aggfunc="first")

        .fillna(0.0)

        .reindex(true_w.index)

    )

    cultivars = true_w.index.tolist()

    vocs = true_w.columns.tolist()

    Yt = true_w.to_numpy(dtype=float)

    Yp = pred_w.to_numpy(dtype=float)

    Yp_fix = np.zeros_like(Yp)

    for i, _c in enumerate(cultivars):

        mask = np.ones(len(cultivars), dtype=bool)

        mask[i] = False

        mu_train = Yt[mask].mean(axis=0)

        Yp_fix[i] = 2.0 * mu_train - Yp[i]

    if args.clip_nonneg:

        Yp_fix = np.clip(Yp_fix, 0.0, None)

    fix_w = pd.DataFrame(Yp_fix, index=cultivars, columns=vocs)

    fix_w.index.name = "cultivar"

    true_w.index.name = "cultivar"

    pred_w.index.name = "cultivar"

    out_long = fix_w.reset_index().melt(id_vars=["cultivar"], var_name="VOC", value_name="y_pred")

    out_true = true_w.reset_index().melt(id_vars=["cultivar"], var_name="VOC", value_name="y_true")

    out_df = out_long.merge(out_true, on=["cultivar", "VOC"], how="left")

    raw_long = pred_w.reset_index().melt(id_vars=["cultivar"], var_name="VOC", value_name="y_pred_raw")

    out_df = out_df.merge(raw_long, on=["cultivar", "VOC"], how="left")

    if has_panel:

        out_df.insert(0, "panel", panel_val)

    out_path = Path(args.out) if args.out else p.parent / (p.stem + "_reflect_loco.parquet")

    if out_path.suffix.lower() == ".parquet":

        out_df.to_parquet(out_path, index=False)

    else:

        out_df.to_csv(out_path, index=False)

    print("[OK] Saved reflected predictions:", out_path)

    print("  rows:", out_df.shape[0], "cols:", out_df.shape[1])

    print("  clip_nonneg:", bool(args.clip_nonneg))

if __name__ == "__main__":

    main()
