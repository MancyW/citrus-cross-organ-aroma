from __future__ import annotations

import argparse

import json

from dataclasses import dataclass

from datetime import datetime

from pathlib import Path

from typing import List, Dict, Tuple

import numpy as np

import pandas as pd

import yaml

from src.fig4.models.hurdle import HurdleRegressor

from src.fig4.cv.loco import iter_loco

from src.fig4.utils.seed import set_global_seed

def read_yaml(p: str | Path) -> dict:

    with Path(p).open("r", encoding="utf-8") as f:

        return yaml.safe_load(f)

def ensure_dir(p: str | Path):

    Path(p).mkdir(parents=True, exist_ok=True)

def parse_panel(s: str) -> List[str]:

    return [x.strip() for x in s.split("+")]

def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:

    na = np.linalg.norm(a) + eps

    nb = np.linalg.norm(b) + eps

    return float(np.dot(a, b) / (na * nb))

def pearson_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:

    a = a.astype(float)

    b = b.astype(float)

    a = a - a.mean()

    b = b - b.mean()

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps

    return float(np.dot(a, b) / denom)

def build_leaf_panel_features(

    df_cs: pd.DataFrame,

    voc_cols: List[str],

    cultivars: List[str],

    stage_col: str,

    cultivar_col: str,

    organ_col: str,

    leaf_label: str,

    stages_panel: List[str],

    fusion_mode: str,

    add_missing_indicators: bool,

) -> Tuple[np.ndarray, List[str]]:

    leaf_df = df_cs[(df_cs[organ_col] == leaf_label)].copy()

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

            mats = []

            miss = []

            for st in stages_panel:

                v = lut.get((c, st), None)

                if v is None:

                    miss.append(1.0)

                else:

                    miss.append(0.0)

                    mats.append(v)

            if len(mats) == 0:

                x = np.zeros(len(voc_cols), dtype=float)

            else:

                x = np.vstack(mats).mean(axis=0)

            if add_missing_indicators:

                x = np.concatenate([x, np.array(miss, dtype=float)])

            X_rows.append(x)

    elif fusion_mode == "concat":

        feat_names = []

        for st in stages_panel:

            feat_names += [f"leaf_{st}_{v}" for v in voc_cols]

        if add_missing_indicators:

            feat_names += [f"miss_{st}" for st in stages_panel]

        for c in cultivars:

            chunks = []

            miss = []

            for st in stages_panel:

                v = lut.get((c, st), None)

                if v is None:

                    miss.append(1.0)

                    chunks.append(np.zeros(len(voc_cols), dtype=float))

                else:

                    miss.append(0.0)

                    chunks.append(v)

            x = np.concatenate(chunks)

            if add_missing_indicators:

                x = np.concatenate([x, np.array(miss, dtype=float)])

            X_rows.append(x)

    else:

        raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    return np.vstack(X_rows), feat_names

def build_leaf_trajectory_stats(

    df_cs: pd.DataFrame,

    voc_cols: List[str],

    cultivars: List[str],

    stage_col: str,

    cultivar_col: str,

    organ_col: str,

    leaf_label: str,

    ordered_stages: List[str],

    stats: List[str],

) -> Tuple[np.ndarray, List[str]]:

    leaf_df = df_cs[(df_cs[organ_col] == leaf_label)].copy()

    stage_index = {st: i for i, st in enumerate(ordered_stages)}

    nS = len(ordered_stages)

    V = len(voc_cols)

    x_axis = np.arange(nS, dtype=float)

    lut = {}

    for _, r in leaf_df.iterrows():

        lut[(r[cultivar_col], r[stage_col])] = r[voc_cols].to_numpy(dtype=float)

    feat_names = []

    for st in stats:

        feat_names += [f"traj_{st}_{v}" for v in voc_cols]

    X_rows = []

    for c in cultivars:

        M = np.full((nS, V), np.nan, dtype=float)

        for st in ordered_stages:

            v = lut.get((c, st), None)

            if v is not None:

                M[stage_index[st], :] = v

        Y = np.nan_to_num(M, nan=0.0)

        feats = []

        if "slope" in stats:

            xm = x_axis.mean()

            vx = ((x_axis - xm) ** 2).sum()

            ym = Y.mean(axis=0)

            cov = ((x_axis[:, None] - xm) * (Y - ym[None, :])).sum(axis=0)

            slope = cov / (vx + 1e-12)

            feats.append(slope)

        if "auc" in stats:

            auc = np.trapz(Y, x_axis, axis=0)

            feats.append(auc)

        if any(s.startswith("delta") for s in stats):

            delta = Y[-1, :] - Y[0, :]

            feats.append(delta)

        if len(feats) == 0:

            X_rows.append(np.zeros(0, dtype=float))

        else:

            X_rows.append(np.concatenate(feats))

    return np.vstack(X_rows), feat_names

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--config", default="configs/fig4/base.yaml")

    args = ap.parse_args()

    cfg = read_yaml(args.config)

    paths = cfg["paths"]

    cols = cfg.get("cols", {})

    cultivar_col = cols.get("cultivar", "Cultivar")

    organ_col = cols.get("organ", "Organ")

    stage_col = cols.get("stage", "Stage")

    stages_cfg = cfg.get("stages", {})

    ordered_stages = stages_cfg.get("ordered", ["S1", "S2", "S3", "S4"])

    target_stage = stages_cfg.get("target_peel_stage", "S4")

    leaf_label = stages_cfg.get("leaf_organ_label", "Leaf")

    peel_label = stages_cfg.get("peel_organ_label", "Peel")

    panels_cfg = cfg.get("panels", {})

    fusion_mode = panels_cfg.get("fusion_mode", "concat")

    add_miss = bool(panels_cfg.get("add_missing_indicators", True))

    panel_list = panels_cfg.get("candidates", ["S1", "S2", "S3", "S4", "S1+S2", "S1+S2+S3+S4"])

    traj_cfg = panels_cfg.get("trajectory", {})

    traj_enabled = bool(traj_cfg.get("enabled", True))

    traj_stats = traj_cfg.get("stats", ["slope", "auc", "delta_S4_S1"])

    model_cfg = cfg.get("model", {})

    random_state = int(model_cfg.get("random_state", 42))

    n_jobs = int(model_cfg.get("n_jobs", 8))

    min_pos = int(model_cfg.get("min_pos", 8))

    log1p = bool(model_cfg.get("log1p", True))

    clf_C = float(model_cfg.get("classifier", {}).get("C", 1.0))

    clf_max_iter = int(model_cfg.get("classifier", {}).get("max_iter", 5000))

    ridge_alpha = float(model_cfg.get("regressor", {}).get("ridge_alpha", 1.0))

    eval_cfg = cfg.get("eval", {})

    topk_fracs = eval_cfg.get("topk_fracs", [0.05, 0.10, 0.20])

    set_global_seed(random_state)

    df_cs = pd.read_parquet(paths["ssot_cultivar_stage"])

    meta_cols = {cultivar_col, stage_col, organ_col, "n_samples"}

    voc_cols = [c for c in df_cs.columns if c not in meta_cols]

    voc_cols = sorted(voc_cols)

    peel_s4 = df_cs[(df_cs[organ_col] == peel_label) & (df_cs[stage_col] == target_stage)].copy()

    cultivars = sorted(peel_s4[cultivar_col].unique().tolist())

    Y = []

    for c in cultivars:

        row = peel_s4[peel_s4[cultivar_col] == c]

        if len(row) != 1:

            raise ValueError(f"Expected exactly 1 peel S4 row for cultivar={c}, got {len(row)}.")

        Y.append(row[voc_cols].iloc[0].to_numpy(dtype=float))

    Y = np.vstack(Y)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    out_dir = Path("results/runs") / run_id

    ensure_dir(out_dir)

    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    all_metrics = []

    pred_rows = []

    groups = np.array(cultivars)

    for panel in panel_list:

        stages_panel = parse_panel(panel)

        X_panel, feat_panel = build_leaf_panel_features(

            df_cs=df_cs,

            voc_cols=voc_cols,

            cultivars=cultivars,

            stage_col=stage_col,

            cultivar_col=cultivar_col,

            organ_col=organ_col,

            leaf_label=leaf_label,

            stages_panel=stages_panel,

            fusion_mode=fusion_mode,

            add_missing_indicators=add_miss,

        )

        X = X_panel

        feat_names = feat_panel

        if traj_enabled:

            X_traj, feat_traj = build_leaf_trajectory_stats(

                df_cs=df_cs,

                voc_cols=voc_cols,

                cultivars=cultivars,

                stage_col=stage_col,

                cultivar_col=cultivar_col,

                organ_col=organ_col,

                leaf_label=leaf_label,

                ordered_stages=ordered_stages,

                stats=traj_stats,

            )

            X = np.hstack([X, X_traj])

            feat_names = feat_names + feat_traj

        fold_cos = []

        fold_corr = []

        true_score = np.log1p(Y).sum(axis=1) if log1p else Y.sum(axis=1)

        for tr_idx, te_idx in iter_loco(groups):

            Xtr, Xte = X[tr_idx], X[te_idx]

            Ytr, Yte = Y[tr_idx], Y[te_idx]

            model = HurdleRegressor(

                min_pos=min_pos,

                log1p=log1p,

                n_jobs=n_jobs,

                random_state=random_state,

                clf_C=clf_C,

                clf_max_iter=clf_max_iter,

                ridge_alpha=ridge_alpha,

            )

            model.fit(Xtr, Ytr)

            Yhat = model.predict(Xte)

            for i_local, i_global in enumerate(te_idx):

                yt = Yte[i_local]

                yp = Yhat[i_local]

                if log1p:

                    yt_eval = np.log1p(yt)

                    yp_eval = np.log1p(yp)

                else:

                    yt_eval, yp_eval = yt, yp

                fold_cos.append(cosine_sim(yp_eval, yt_eval))

                fold_corr.append(pearson_corr(yp_eval, yt_eval))

                pred_rows.append({

                    "run_id": run_id,

                    "panel": panel,

                    "cultivar": cultivars[i_global],

                    "cosine": cosine_sim(yp_eval, yt_eval),

                    "pearson": pearson_corr(yp_eval, yt_eval),

                    "true_score": float(true_score[i_global]),

                    "pred_score": float(np.log1p(yp).sum() if log1p else yp.sum()),

                })

        pred_score = np.array([r["pred_score"] for r in pred_rows if r["panel"] == panel])

        true_score_panel = np.array([r["true_score"] for r in pred_rows if r["panel"] == panel])

        metrics = {

            "panel": panel,

            "fusion_mode": fusion_mode,

            "traj_enabled": traj_enabled,

            "traj_stats": "+".join(traj_stats) if traj_enabled else "",

            "n_cultivars": len(cultivars),

            "n_vocs": len(voc_cols),

            "n_features": int(X.shape[1]),

            "mean_cosine": float(np.mean(fold_cos)),

            "std_cosine": float(np.std(fold_cos)),

            "mean_pearson": float(np.mean(fold_corr)),

            "std_pearson": float(np.std(fold_corr)),

        }

        for frac in topk_fracs:

            k = max(1, int(round(len(cultivars) * float(frac))))

            top_pred = np.argsort(-pred_score)[:k]

            top_true = np.argsort(-true_score_panel)[:k]

            recall = len(set(top_pred).intersection(set(top_true))) / float(k)

            enrich = (true_score_panel[top_pred].mean() + 1e-12) / (true_score_panel.mean() + 1e-12)

            metrics[f"top{int(frac*100)}_recall"] = float(recall)

            metrics[f"top{int(frac*100)}_enrichment"] = float(enrich)

        all_metrics.append(metrics)

        print(f"[PANEL] {panel} | mean_cos={metrics['mean_cosine']:.3f} | mean_r={metrics['mean_pearson']:.3f}")

    metrics_df = pd.DataFrame(all_metrics).sort_values(["mean_cosine", "mean_pearson"], ascending=False)

    metrics_df.to_csv(out_dir / "panel_metrics.csv", index=False)

    preds_df = pd.DataFrame(pred_rows)

    preds_df.to_csv(out_dir / "predictions_summary.csv", index=False)

    best = metrics_df.iloc[0].to_dict()

    card = {

        "run_id": run_id,

        "best_panel": best.get("panel"),

        "best_mean_cosine": best.get("mean_cosine"),

        "best_mean_pearson": best.get("mean_pearson"),

        "n_cultivars": best.get("n_cultivars"),

        "n_vocs": best.get("n_vocs"),

        "notes": [

            "Evaluation: LOCO (Leave-One-Cultivar-Out)",

            "Target: Peel S4 VOC profile",

            "Model: Two-part (Hurdle) per VOC; structural zero if positives < min_pos",

        ],

    }

    (out_dir / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[OK] Saved:")

    print("  -", str(out_dir / "panel_metrics.csv"))

    print("  -", str(out_dir / "predictions_summary.csv"))

    print("  -", str(out_dir / "model_card.json"))

if __name__ == "__main__":

    main()
