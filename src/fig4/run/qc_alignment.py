from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--pred_long", required=True, help="pred_vectors_long.parquet or .csv")

    ap.add_argument("--outdir", default=None)

    args = ap.parse_args()

    p = Path(args.pred_long)

    outdir = Path(args.outdir) if args.outdir else p.parent / "qc_alignment"

    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)

    print("[QC] rows:", df.shape[0], "cols:", df.shape[1])

    print("[QC] n cultivars:", df["cultivar"].nunique(), "n VOC:", df["VOC"].nunique())

    dup = df.duplicated(subset=["cultivar", "VOC"]).sum()

    print("[QC] duplicated (cultivar,VOC) rows:", int(dup))

    vocs = sorted(df["VOC"].unique().tolist())

    missing_rows = []

    for c, g in df.groupby("cultivar"):

        miss = sorted(set(vocs) - set(g["VOC"].unique().tolist()))

        if miss:

            missing_rows.append({

                "cultivar": c,

                "n_missing": len(miss),

                "example_missing": ";".join(miss[:5])

            })

    if len(missing_rows) == 0:

        miss_df = pd.DataFrame(columns=["cultivar", "n_missing", "example_missing"])

    else:

        miss_df = pd.DataFrame(missing_rows).sort_values("n_missing", ascending=False)

    miss_df.to_csv(outdir / "missing_voc_by_cultivar.csv", index=False)

    print("[QC] cultivars with missing VOC:", int(miss_df.shape[0]))

    corr_rows = []

    for voc, g in df.groupby("VOC"):

        a = g["y_pred"].to_numpy(dtype=float)

        b = g["y_true"].to_numpy(dtype=float)

        if np.std(a) < 1e-12 or np.std(b) < 1e-12:

            r = np.nan

        else:

            r = float(np.corrcoef(a, b)[0, 1])

        corr_rows.append({"VOC": voc, "corr_pred_true_across_cultivars": r,

                          "pred_std": float(np.std(a)), "true_std": float(np.std(b))})

    corr_df = pd.DataFrame(corr_rows).sort_values("corr_pred_true_across_cultivars")

    corr_df.to_csv(outdir / "vocwise_corr_across_cultivars.csv", index=False)

    print("[QC] VOC-wise corr (across cultivars):")

    print("  median:", float(np.nanmedian(corr_df["corr_pred_true_across_cultivars"].to_numpy())))

    print("  10th pct:", float(np.nanpercentile(corr_df["corr_pred_true_across_cultivars"].to_numpy(), 10)))

    print("  90th pct:", float(np.nanpercentile(corr_df["corr_pred_true_across_cultivars"].to_numpy(), 90)))

    cv_rows = []

    for c, g in df.groupby("cultivar"):

        a = np.log1p(g["y_pred"].to_numpy(dtype=float))

        b = np.log1p(g["y_true"].to_numpy(dtype=float))

        if np.std(a) < 1e-12 or np.std(b) < 1e-12:

            r = np.nan

        else:

            r = float(np.corrcoef(a, b)[0, 1])

        cv_rows.append({"cultivar": c, "corr_across_VOC_log1p": r,

                        "pred_total_logsum": float(np.log1p(g["y_pred"].to_numpy(dtype=float)).sum()),

                        "true_total_logsum": float(np.log1p(g["y_true"].to_numpy(dtype=float)).sum())})

    cv_df = pd.DataFrame(cv_rows).sort_values("corr_across_VOC_log1p")

    cv_df.to_csv(outdir / "cultivarwise_corr_across_VOC.csv", index=False)

    print("[QC] cultivar-wise corr across VOC (log1p):")

    print("  min/median/max:", float(np.nanmin(cv_df["corr_across_VOC_log1p"])),

          float(np.nanmedian(cv_df["corr_across_VOC_log1p"])),

          float(np.nanmax(cv_df["corr_across_VOC_log1p"])))

    print("[OK] Saved QC reports to:", outdir)

if __name__ == "__main__":

    main()
