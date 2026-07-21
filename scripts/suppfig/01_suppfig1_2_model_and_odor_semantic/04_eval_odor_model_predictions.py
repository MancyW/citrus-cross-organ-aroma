#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

def safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)

def safe_ap(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.sum() == 0:
        return np.nan
    return average_precision_score(y_true, y_score)

def load_from_pred_csv(pred_csv, true_prefix="y_true__", pred_prefix="y_pred__"):
    df = pd.read_csv(pred_csv)
    true_cols = [c for c in df.columns if c.startswith(true_prefix)]
    pred_cols = [c for c in df.columns if c.startswith(pred_prefix)]

    if len(true_cols) == 0 or len(pred_cols) == 0:
        raise ValueError(
            f"Cannot find columns with prefixes {true_prefix} and {pred_prefix} in {pred_csv}"
        )

    desc_true = [c[len(true_prefix):] for c in true_cols]
    desc_pred = [c[len(pred_prefix):] for c in pred_cols]

    if desc_true != desc_pred:
        raise ValueError("Descriptor names inferred from y_true__ and y_pred__ do not match.")

    y_true = df[true_cols].to_numpy(dtype=float)
    y_pred = df[pred_cols].to_numpy(dtype=float)
    return y_true, y_pred, desc_true, df

def load_from_npy(true_npy, pred_npy, desc_names_file):
    y_true = np.load(true_npy)
    y_pred = np.load(pred_npy)

    if desc_names_file.endswith(".txt"):
        with open(desc_names_file, "r", encoding="utf-8") as f:
            desc_names = [x.strip() for x in f if x.strip()]
    else:
        tmp = pd.read_csv(desc_names_file)
        if tmp.shape[1] == 1:
            desc_names = tmp.iloc[:, 0].astype(str).tolist()
        else:
            desc_names = tmp["descriptor"].astype(str).tolist()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred have different shapes.")
    if y_true.shape[1] != len(desc_names):
        raise ValueError("Number of descriptor names does not match y_true/y_pred columns.")
    return y_true, y_pred, desc_names, None

def compute_metrics(y_true, y_pred, desc_names, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    y_bin = (y_pred >= threshold).astype(int)

    per_desc = []
    for i, desc in enumerate(desc_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        yb = y_bin[:, i]

        auroc = safe_auroc(yt, yp)
        ap = safe_ap(yt, yp)
        f1 = f1_score(yt, yb, zero_division=0)
        prec = precision_score(yt, yb, zero_division=0)
        rec = recall_score(yt, yb, zero_division=0)

        per_desc.append({
            "descriptor": desc,
            "support_positive": int(yt.sum()),
            "prevalence": float(yt.mean()),
            "auroc": auroc,
            "average_precision": ap,
            "f1_at_0.5": f1,
            "precision_at_0.5": prec,
            "recall_at_0.5": rec,
        })

    per_desc_df = pd.DataFrame(per_desc).sort_values(
        by=["average_precision", "auroc"], ascending=False
    )

    macro_auroc = np.nanmean(per_desc_df["auroc"].to_numpy())
    macro_ap = np.nanmean(per_desc_df["average_precision"].to_numpy())
    macro_f1 = np.nanmean(per_desc_df["f1_at_0.5"].to_numpy())

    # micro metrics
    yt_flat = y_true.ravel()
    yp_flat = y_pred.ravel()
    yb_flat = y_bin.ravel()

    micro_auroc = safe_auroc(yt_flat, yp_flat)
    micro_ap = safe_ap(yt_flat, yp_flat)
    micro_f1 = f1_score(yt_flat, yb_flat, zero_division=0)
    micro_precision = precision_score(yt_flat, yb_flat, zero_division=0)
    micro_recall = recall_score(yt_flat, yb_flat, zero_division=0)

    exact_match = float((y_true == y_bin).all(axis=1).mean())
    subset_accuracy = exact_match
    label_density = float(y_true.sum(axis=1).mean())
    n_samples, n_labels = y_true.shape

    summary_df = pd.DataFrame([{
        "n_test_samples": n_samples,
        "n_descriptors": n_labels,
        "threshold_for_binary_metrics": threshold,
        "macro_auroc": macro_auroc,
        "macro_average_precision": macro_ap,
        "macro_f1_at_0.5": macro_f1,
        "micro_auroc": micro_auroc,
        "micro_average_precision": micro_ap,
        "micro_f1_at_0.5": micro_f1,
        "micro_precision_at_0.5": micro_precision,
        "micro_recall_at_0.5": micro_recall,
        "subset_accuracy_exact_match": subset_accuracy,
        "mean_positive_labels_per_sample": label_density,
    }])

    return summary_df, per_desc_df

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate held-out performance of the odor-descriptor model on the public odor dataset."
    )
    parser.add_argument("--pred_csv", type=str, default=None,
                        help="CSV containing y_true__* and y_pred__* columns.")
    parser.add_argument("--true_npy", type=str, default=None,
                        help="Numpy file for true labels if pred_csv is not used.")
    parser.add_argument("--pred_npy", type=str, default=None,
                        help="Numpy file for predicted probabilities if pred_csv is not used.")
    parser.add_argument("--desc_names", type=str, default=None,
                        help="Descriptor names file (.txt or .csv) if npy inputs are used.")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--true_prefix", type=str, default="y_true__")
    parser.add_argument("--pred_prefix", type=str, default="y_pred__")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.pred_csv is not None:
        y_true, y_pred, desc_names, raw_df = load_from_pred_csv(
            args.pred_csv,
            true_prefix=args.true_prefix,
            pred_prefix=args.pred_prefix,
        )
    else:
        if args.true_npy is None or args.pred_npy is None or args.desc_names is None:
            raise ValueError("Either --pred_csv or all of --true_npy --pred_npy --desc_names must be provided.")
        y_true, y_pred, desc_names, raw_df = load_from_npy(
            args.true_npy, args.pred_npy, args.desc_names
        )

    summary_df, per_desc_df = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        desc_names=desc_names,
        threshold=args.threshold,
    )

    summary_path = outdir / "SupplementaryTable_OdorModelHeldOut_Summary.csv"
    per_desc_path = outdir / "SupplementaryTable_OdorModelHeldOut_PerDescriptor.csv"
    txt_path = outdir / "odor_model_heldout_summary.txt"

    summary_df.to_csv(summary_path, index=False)
    per_desc_df.to_csv(per_desc_path, index=False)

    s = summary_df.iloc[0]
    text = (
        f"Held-out performance on the public odor dataset (n={int(s['n_test_samples'])} test samples; "
        f"{int(s['n_descriptors'])} descriptors): "
        f"macro-AUROC={s['macro_auroc']:.4f}, "
        f"macro-AP={s['macro_average_precision']:.4f}, "
        f"macro-F1@0.5={s['macro_f1_at_0.5']:.4f}; "
        f"micro-AUROC={s['micro_auroc']:.4f}, "
        f"micro-AP={s['micro_average_precision']:.4f}, "
        f"micro-F1@0.5={s['micro_f1_at_0.5']:.4f}."
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"[OK] wrote: {summary_path}")
    print(f"[OK] wrote: {per_desc_path}")
    print(f"[OK] wrote: {txt_path}")
    print(text)

if __name__ == "__main__":
    main()