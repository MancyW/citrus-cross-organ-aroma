from __future__ import annotations

import argparse

from datetime import datetime

from pathlib import Path

from typing import List

import numpy as np

import pandas as pd

import yaml

from src.models.hurdle import HurdleRegressor

from src.cv.loco import iter_loco

from src.utils.seed import set_global_seed

def read_yaml(p: str | Path) -> dict:

    with Path(p).open("r", encoding="utf-8") as f:

        return yaml.safe_load(f)

def ensure_dir(p: str | Path):

    Path(p).mkdir(parents=True, exist_ok=True)

def parse_panel(s: str) -> List[str]:

    return [x.strip() for x in s.split("+")]

def build_X_leaf_singlepanel(

    df_cs: pd.DataFrame,

    voc_cols: List[str],

    cultivars: List[str],

    panel: str,

    fusion_mode: str = "concat",

    add_missing_indicators: bool = True,

    cultivar_col: str = "Cultivar",

    stage_col: str = "Stage",

    organ_col: str = "Organ",

    leaf_label: str = "Leaf",

):

    stages_panel = parse_panel(panel)

    leaf_df = df_cs[df_cs[organ_col] == leaf_label].copy()

    lut = {}

    for _, r in leaf_df.iterrows():

        lut[(r[cultivar_col], r[stage_col])] = r[voc_cols].to_numpy(dtype=float)

    X_rows = []

    feat_names = []

    if fusion_mode == "mean":

        feat_names = [f"leaf_mean_{v}" for v in voc_cols]

        if add_missing_indicators:

            feat_names += [f"miss_{st}" for st in stages_panel]

        for c in cultivars:

            mats, miss = [], []

            for st in stages_panel:

                v = lut.get((c, st), None)

                if v is None:

                    miss.append(1.0)

                else:

                    miss.append(0.0)

                    mats.append(v)

            x = np.vstack(mats).mean(axis=0) if len(mats) else np.zeros(len(voc_cols))

            if add_missing_indicators:

                x = np.concatenate([x, np.array(miss)])

            X_rows.append(x)

    elif fusion_mode == "concat":

        feat_names = []

        for st in stages_panel:

            feat_names += [f"leaf_{st}_{v}" for v in voc_cols]

        if add_missing_indicators:

            feat_names += [f"miss_{st}" for st in stages_panel]

        for c in cultivars:

            chunks, miss = [], []

            for st in stages_panel:

                v = lut.get((c, st), None)

                if v is None:

                    miss.append(1.0)

                    chunks.append(np.zeros(len(voc_cols)))

                else:

                    miss.append(0.0)

                    chunks.append(v)

            x = np.concatenate(chunks)

            if add_missing_indicators:

                x = np.concatenate([x, np.array(miss)])

            X_rows.append(x)

    else:

        raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    return np.vstack(X_rows), feat_names

def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:

    na = np.linalg.norm(a) + eps

    nb = np.linalg.norm(b) + eps

    return float(np.dot(a, b) / (na * nb))

def pearson_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:

    a = a - a.mean()

    b = b - b.mean()

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps

    return float(np.dot(a, b) / denom)

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--config", default="configs/base.yaml")

    ap.add_argument("--panel", default="S1", help="e.g., S1 or S1+S2")

    ap.add_argument("--outdir", default=None)

    args = ap.parse_args()

    cfg = read_yaml(args.config)

    set_global_seed(int(cfg.get("model", {}).get("random_state", 42)))

    cols = cfg.get("cols", {})

    cultivar_col = cols.get("cultivar", "Cultivar")

    stage_col = cols.get("stage", "Stage")

    organ_col = cols.get("organ", "Organ")

    stages_cfg = cfg.get("stages", {})

    target_stage = stages_cfg.get("target_peel_stage", "S4")

    leaf_label = stages_cfg.get("leaf_organ_label", "Leaf")

    peel_label = stages_cfg.get("peel_organ_label", "Peel")

    panels_cfg = cfg.get("panels", {})

    fusion_mode = panels_cfg.get("fusion_mode", "concat")

    add_miss = bool(panels_cfg.get("add_missing_indicators", True))

    model_cfg = cfg.get("model", {})

    n_jobs = int(model_cfg.get("n_jobs", 8))

    min_pos = int(model_cfg.get("min_pos", 8))

    log1p = bool(model_cfg.get("log1p", True))

    clf_C = float(model_cfg.get("classifier", {}).get("C", 1.0))

    clf_max_iter = int(model_cfg.get("classifier", {}).get("max_iter", 5000))

    ridge_alpha = float(model_cfg.get("regressor", {}).get("ridge_alpha", 1.0))

    random_state = int(model_cfg.get("random_state", 42))

    paths = cfg["paths"]

    df_cs = pd.read_parquet(paths["ssot_cultivar_stage"])

    meta_cols = {cultivar_col, stage_col, organ_col, "n_samples"}

    voc_cols = sorted([c for c in df_cs.columns if c not in meta_cols])

    peel_s4 = df_cs[(df_cs[organ_col] == peel_label) & (df_cs[stage_col] == target_stage)].copy()

    cultivars = sorted(peel_s4[cultivar_col].unique().tolist())

    Y_true = []

    for c in cultivars:

        row = peel_s4[peel_s4[cultivar_col] == c]

        Y_true.append(row[voc_cols].iloc[0].to_numpy(dtype=float))

    Y_true = np.vstack(Y_true)

    X, feat_names = build_X_leaf_singlepanel(

        df_cs=df_cs, voc_cols=voc_cols, cultivars=cultivars, panel=args.panel,

        fusion_mode=fusion_mode, add_missing_indicators=add_miss,

        cultivar_col=cultivar_col, stage_col=stage_col, organ_col=organ_col, leaf_label=leaf_label

    )

    groups = np.array(cultivars)

    Y_pred = np.zeros_like(Y_true, dtype=float)

    for tr_idx, te_idx in iter_loco(groups):

        Xtr, Xte = X[tr_idx], X[te_idx]

        Ytr = Y_true[tr_idx]

        m = HurdleRegressor(

            min_pos=min_pos, log1p=log1p, n_jobs=n_jobs,

            random_state=random_state, clf_C=clf_C, clf_max_iter=clf_max_iter,

            ridge_alpha=ridge_alpha

        )

        m.fit(Xtr, Ytr)

        yhat = m.predict(Xte)

        Y_pred[te_idx[0], :] = yhat[0, :]

    rows = []

    for i, c in enumerate(cultivars):

        yt = Y_true[i]

        yp = Y_pred[i]

        yt_eval = np.log1p(yt) if log1p else yt

        yp_eval = np.log1p(yp) if log1p else yp

        rows.append({

            "panel": args.panel,

            "cultivar": c,

            "cosine_log": cosine_sim(yp_eval, yt_eval),

            "pearson_log": pearson_corr(yp_eval, yt_eval),

            "total_true_logsum": float(np.log1p(yt).sum()) if log1p else float(yt.sum()),

            "total_pred_logsum": float(np.log1p(yp).sum()) if log1p else float(yp.sum()),

        })

    metrics_df = pd.DataFrame(rows)

    pred_df = pd.DataFrame(Y_pred, columns=voc_cols)

    true_df = pd.DataFrame(Y_true, columns=voc_cols)

    pred_df.insert(0, "cultivar", cultivars)

    true_df.insert(0, "cultivar", cultivars)

    long_pred = pred_df.melt(id_vars=["cultivar"], var_name="VOC", value_name="y_pred")

    long_true = true_df.melt(id_vars=["cultivar"], var_name="VOC", value_name="y_true")

    long = long_pred.merge(long_true, on=["cultivar", "VOC"], how="left")

    long.insert(0, "panel", args.panel)

    ts = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    outdir = Path(args.outdir) if args.outdir else Path("results/pred_vectors") / f"{ts}_{args.panel.replace('+','_')}"

    ensure_dir(outdir)

    metrics_df.to_csv(outdir / "cultivar_metrics.csv", index=False)

    long.to_parquet(outdir / "pred_vectors_long.parquet", index=False)

    long.to_csv(outdir / "pred_vectors_long.csv", index=False)

    (outdir / "feature_names.txt").write_text("\n".join(feat_names), encoding="utf-8")

    print("[OK] Saved:")

    print("  -", outdir / "cultivar_metrics.csv")

    print("  -", outdir / "pred_vectors_long.parquet")

    print("  -", outdir / "pred_vectors_long.csv")

    print("  -", outdir / "feature_names.txt")

if __name__ == "__main__":

    main()
