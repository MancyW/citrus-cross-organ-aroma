from __future__ import annotations

import argparse

import glob

import json

from dataclasses import dataclass

from datetime import datetime

from pathlib import Path

from typing import Dict, List, Optional, Tuple

import numpy as np

import pandas as pd

def _now_tag() -> str:

    return datetime.now().strftime("report_%Y%m%d_%H%M%S")

def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:

    na = np.linalg.norm(a) + eps

    nb = np.linalg.norm(b) + eps

    return float(np.dot(a, b) / (na * nb))

def pearson(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:

    a = a - a.mean()

    b = b - b.mean()

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps

    return float(np.dot(a, b) / denom)

def spearman(x: np.ndarray, y: np.ndarray) -> float:

    rx = pd.Series(x).rank(method="average").to_numpy()

    ry = pd.Series(y).rank(method="average").to_numpy()

    rx = rx - rx.mean()

    ry = ry - ry.mean()

    denom = (np.linalg.norm(rx) * np.linalg.norm(ry)) + 1e-12

    return float(np.dot(rx, ry) / denom)

def to_comp(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:

    v = np.asarray(v, dtype=float)

    v = np.clip(v, 0.0, None)

    s = v.sum()

    if s <= 0:

        return np.full_like(v, 1.0 / len(v))

    p = v / s

    p = np.clip(p, eps, None)

    return p / p.sum()

def clr_from_abs(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:

    p = to_comp(v, eps=eps)

    lp = np.log(np.clip(p, eps, None))

    return lp - lp.mean()

def pairwise_dist_stats(C: np.ndarray) -> Tuple[float, float, float]:

    dists = []

    n = C.shape[0]

    for i in range(n):

        for j in range(i + 1, n):

            dists.append(float(np.linalg.norm(C[i] - C[j])))

    if not dists:

        return 0.0, 0.0, 0.0

    d = np.array(dists, dtype=float)

    return float(d.min()), float(np.median(d)), float(d.max())

def vocwise_corr(P: np.ndarray, T: np.ndarray) -> np.ndarray:

    cors = []

    for j in range(P.shape[1]):

        a, b = P[:, j], T[:, j]

        if np.std(a) < 1e-12 or np.std(b) < 1e-12:

            continue

        cors.append(float(np.corrcoef(a, b)[0, 1]))

    return np.array(cors, dtype=float)

def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float]:

    rng = np.random.default_rng(seed)

    values = np.asarray(values, dtype=float)

    if values.size == 0:

        return 0.0, 0.0, 0.0

    boots = []

    for _ in range(int(n_boot)):

        idx = rng.integers(0, values.size, size=values.size)

        boots.append(float(np.mean(values[idx])))

    boots = np.array(boots, dtype=float)

    return float(np.mean(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def permutation_test_mean_cosine(

    P: np.ndarray,

    T: np.ndarray,

    n_perm: int = 300,

    seed: int = 0,

) -> Tuple[float, float]:

    rng = np.random.default_rng(seed)

    obs = float(np.mean([cosine(P[i], T[i]) for i in range(T.shape[0])]))

    null = []

    for _ in range(int(n_perm)):

        perm = rng.permutation(T.shape[0])

        Tp = T[perm, :]

        null.append(float(np.mean([cosine(P[i], Tp[i]) for i in range(T.shape[0])])))

    null = np.array(null, dtype=float)

    p = float((np.sum(null >= obs) + 1.0) / (null.size + 1.0))

    return obs, p

def build_ideotype(true_w: pd.DataFrame, mode: str, anchor: Optional[str], top_n: int) -> np.ndarray:

    M = true_w.to_numpy(dtype=float)

    if mode == "cultivar":

        if anchor is None or anchor not in true_w.index:

            raise ValueError(f"anchor '{anchor}' not in cultivars")

        return true_w.loc[anchor].to_numpy(dtype=float)

    totals = np.log1p(M).sum(axis=1)

    if mode == "mean_top_total":

        idx = np.argsort(-totals)[: max(1, int(top_n))]

        return M[idx].mean(axis=0)

    if mode == "mean_all":

        return M.mean(axis=0)

    raise ValueError(f"Unknown ideotype_mode: {mode}")

def ideotype_eval(

    W_pred_abs: pd.DataFrame,

    W_true_abs: pd.DataFrame,

    ideotype_mode: str,

    anchor: Optional[str],

    top_n: int,

    eps: float = 1e-6,

) -> Dict[str, float]:

    ide_abs = build_ideotype(W_true_abs, ideotype_mode, anchor, top_n)

    preds, trues, names = [], [], []

    for c in W_true_abs.index:

        yt = W_true_abs.loc[c].to_numpy(float)

        yp = W_pred_abs.loc[c].to_numpy(float)

        ui = clr_from_abs(ide_abs, eps=eps)

        ut = clr_from_abs(yt, eps=eps)

        up = clr_from_abs(yp, eps=eps)

        trues.append(cosine(ut, ui))

        preds.append(cosine(up, ui))

        names.append(c)

    preds = np.array(preds, float)

    trues = np.array(trues, float)

    df = pd.DataFrame({"cultivar": names, "pred_sim": preds, "true_sim": trues})

    def topk_recall(k: int) -> float:

        k = max(1, min(int(k), len(df)))

        top_pred = df.sort_values("pred_sim", ascending=False).head(k)["cultivar"].tolist()

        top_true = df.sort_values("true_sim", ascending=False).head(k)["cultivar"].tolist()

        return len(set(top_pred).intersection(set(top_true))) / float(k)

    return {

        "ideotype_spearman": spearman(preds, trues),

        "ideotype_top1_recall": topk_recall(1),

        "ideotype_top3_recall": topk_recall(3),

        "ideotype_top5_recall": topk_recall(5),

        "pred_sim_std": float(np.std(preds)),

        "pred_sim_range": float(np.max(preds) - np.min(preds)),

    }

def threshold_by_true_quantile_long(

    df_long: pd.DataFrame,

    quantile: float,

    min_pos: int = 1,

    space: str = "log1p",

) -> pd.DataFrame:

    out = df_long.copy()

    if space != "log1p":

        raise ValueError("Only space=log1p supported in this report script")

    thr = {}

    for voc, g in out.groupby("VOC"):

        yt = g["y_true"].to_numpy(float)

        yt_pos = yt[yt > 0]

        if yt_pos.size < int(min_pos):

            thr[voc] = np.inf

        else:

            thr[voc] = float(np.quantile(np.log1p(yt_pos), quantile))

    ypl = np.log1p(out["y_pred"].to_numpy(float))

    thv = out["VOC"].map(thr).to_numpy(float)

    keep = ypl >= thv

    out.loc[~keep, "y_pred"] = 0.0

    return out

@dataclass

class PanelEvalConfig:

    ideotype_mode: str

    anchor: Optional[str]

    top_n: int

    clr_eps: float

    n_boot: int

    boot_seed: int

    n_perm: int

    perm_seed: int

def eval_one_long(df_long: pd.DataFrame, cfg: PanelEvalConfig, do_perm: bool = True) -> Dict[str, float]:

    need = {"cultivar", "VOC", "y_pred", "y_true"}

    miss = need - set(df_long.columns)

    if miss:

        raise KeyError(f"Missing columns: {sorted(miss)}")

    W_pred = (

        df_long.pivot_table(index="cultivar", columns="VOC", values="y_pred", aggfunc="first")

        .fillna(0.0)

        .sort_index()

    )

    W_true = (

        df_long.pivot_table(index="cultivar", columns="VOC", values="y_true", aggfunc="first")

        .fillna(0.0)

        .reindex(W_pred.index)

    )

    P = np.log1p(W_pred.to_numpy(float))

    T = np.log1p(W_true.to_numpy(float))

    n = T.shape[0]

    cos_i = np.array([cosine(P[i], T[i]) for i in range(n)], float)

    r_i = np.array([pearson(P[i], T[i]) for i in range(n)], float)

    mean_cos, lo_cos, hi_cos = bootstrap_ci(cos_i, n_boot=cfg.n_boot, seed=cfg.boot_seed)

    mean_r, lo_r, hi_r = bootstrap_ci(r_i, n_boot=cfg.n_boot, seed=cfg.boot_seed + 7)

    mu = T.mean(axis=0, keepdims=True)

    P_base = np.repeat(mu, n, axis=0)

    base_cos = float(np.mean([cosine(P_base[i], T[i]) for i in range(n)]))

    base_r = float(np.mean([pearson(P_base[i], T[i]) for i in range(n)]))

    C_pred = np.vstack([clr_from_abs(W_pred.iloc[i].to_numpy(float), eps=cfg.clr_eps) for i in range(n)])

    C_true = np.vstack([clr_from_abs(W_true.iloc[i].to_numpy(float), eps=cfg.clr_eps) for i in range(n)])

    pred_dmin, pred_dmed, pred_dmax = pairwise_dist_stats(C_pred)

    true_dmin, true_dmed, true_dmax = pairwise_dist_stats(C_true)

    cors = vocwise_corr(P, T)

    cors_n = int(cors.size)

    cors_med = float(np.median(cors)) if cors_n else 0.0

    cors_frac_neg = float((cors < 0).mean()) if cors_n else 0.0

    cors_p10 = float(np.percentile(cors, 10)) if cors_n else 0.0

    cors_p90 = float(np.percentile(cors, 90)) if cors_n else 0.0

    pred_frac_pos = float((W_pred.to_numpy(float) > 0).mean())

    true_frac_pos = float((W_true.to_numpy(float) > 0).mean())

    spars_ratio = float(pred_frac_pos / (true_frac_pos + 1e-12))

    ide = ideotype_eval(

        W_pred_abs=W_pred,

        W_true_abs=W_true,

        ideotype_mode=cfg.ideotype_mode,

        anchor=cfg.anchor,

        top_n=cfg.top_n,

        eps=cfg.clr_eps,

    )

    out = {

        "n_cultivars": int(n),

        "n_vocs": int(T.shape[1]),

        "mean_cosine": mean_cos,

        "ci_cosine_lo": lo_cos,

        "ci_cosine_hi": hi_cos,

        "mean_pearson": mean_r,

        "ci_pearson_lo": lo_r,

        "ci_pearson_hi": hi_r,

        "baseline_global_mean_cosine": base_cos,

        "baseline_global_mean_pearson": base_r,

        "delta_cosine_vs_baseline": float(mean_cos - base_cos),

        "pred_frac_pos": pred_frac_pos,

        "true_frac_pos": true_frac_pos,

        "sparsity_ratio": spars_ratio,

        "voc_corr_n": cors_n,

        "voc_corr_median": cors_med,

        "voc_corr_p10": cors_p10,

        "voc_corr_p90": cors_p90,

        "voc_corr_frac_neg": cors_frac_neg,

        "pred_clr_dist_min": pred_dmin,

        "pred_clr_dist_median": pred_dmed,

        "pred_clr_dist_max": pred_dmax,

        "true_clr_dist_min": true_dmin,

        "true_clr_dist_median": true_dmed,

        "true_clr_dist_max": true_dmax,

        "clr_dist_ratio_median": float(pred_dmed / (true_dmed + 1e-12)),

        **ide,

    }

    if do_perm and cfg.n_perm > 0:

        obs, pval = permutation_test_mean_cosine(P, T, n_perm=cfg.n_perm, seed=cfg.perm_seed)

        out["perm_obs_mean_cosine"] = obs

        out["perm_pvalue_mean_cosine"] = pval

    return out

def find_latest_pred_long(patterns: List[str]) -> Optional[str]:

    cands = []

    for pat in patterns:

        cands += glob.glob(pat)

    cands = sorted(cands)

    return cands[-1] if cands else None

def panel_to_token(panel: str) -> str:

    return panel.replace("+", "_")

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--panels", default="S1,S2,S3,S4,S1+S2,S1+S2+S3+S4")

    ap.add_argument("--variant", default="LOCO", choices=["LOCO", "FITALL", "BOTH"])

    ap.add_argument("--outdir", default=None)

    ap.add_argument("--ideotype_mode", default="cultivar", choices=["cultivar", "mean_top_total", "mean_all"])

    ap.add_argument("--anchor", default="MTH")

    ap.add_argument("--top_n", type=int, default=3)

    ap.add_argument("--clr_eps", type=float, default=1e-6)

    ap.add_argument("--n_boot", type=int, default=2000)

    ap.add_argument("--n_perm", type=int, default=300)

    ap.add_argument("--threshold_quantiles", default="0,0.05")

    ap.add_argument("--threshold_space", default="log1p", choices=["log1p"])

    ap.add_argument("--threshold_min_pos", type=int, default=1)

    args = ap.parse_args()

    panels = [x.strip() for x in args.panels.split(",") if x.strip()]

    q_list = [float(x.strip()) for x in args.threshold_quantiles.split(",") if x.strip()]

    outdir = Path(args.outdir) if args.outdir else Path("results/robustness") / _now_tag()

    outdir.mkdir(parents=True, exist_ok=True)

    cfg = PanelEvalConfig(

        ideotype_mode=args.ideotype_mode,

        anchor=args.anchor,

        top_n=int(args.top_n),

        clr_eps=float(args.clr_eps),

        n_boot=int(args.n_boot),

        boot_seed=0,

        n_perm=int(args.n_perm),

        perm_seed=0,

    )

    rows = []

    missing = []

    for panel in panels:

        token = panel_to_token(panel)

        want_loco = args.variant in ("LOCO", "BOTH")

        want_fitall = args.variant in ("FITALL", "BOTH")

        loco_path = None

        fitall_path = None

        if want_loco:

            loco_path = find_latest_pred_long([

                f"results/pred_vectors/run_*_{token}/pred_vectors_long.parquet",

            ])

        if want_fitall:

            fitall_path = find_latest_pred_long([

                f"results/pred_vectors/run_*_{token}_FITALL/pred_vectors_long.parquet",

            ])

        for tag, pth in [("LOCO", loco_path), ("FITALL", fitall_path)]:

            if pth is None:

                if (tag == "LOCO" and want_loco) or (tag == "FITALL" and want_fitall):

                    missing.append({"panel": panel, "variant": tag, "reason": "file_not_found"})

                continue

            df = pd.read_parquet(pth)

            met = eval_one_long(df, cfg, do_perm=True)

            met.update({"panel": panel, "variant": tag, "postprocess": "raw", "source": pth})

            rows.append(met)

            for q in q_list:

                if q <= 0:

                    continue

                df_th = threshold_by_true_quantile_long(

                    df_long=df,

                    quantile=q,

                    min_pos=args.threshold_min_pos,

                    space=args.threshold_space,

                )

                met_th = eval_one_long(df_th, cfg, do_perm=False)

                met_th.update({

                    "panel": panel,

                    "variant": tag,

                    "postprocess": f"thresh_q{q:g}_{args.threshold_space}",

                    "source": pth,

                })

                rows.append(met_th)

    rep = pd.DataFrame(rows)

    sel = None

    if not rep.empty and (rep["variant"] == "LOCO").any():

        loco = rep[rep["variant"] == "LOCO"].copy()

        loco["score"] = loco["mean_cosine"] - 0.05 * np.abs(np.log(loco["sparsity_ratio"] + 1e-12))

        best = loco.sort_values("score", ascending=False).head(1)

        if len(best):

            sel = best.iloc[0].to_dict()

    rep_path = outdir / "panel_report.csv"

    rep.to_csv(rep_path, index=False)

    meta = {

        "created_at": datetime.now().isoformat(timespec="seconds"),

        "panels": panels,

        "variant": args.variant,

        "threshold_quantiles": q_list,

        "ideotype": {

            "mode": args.ideotype_mode,

            "anchor": args.anchor,

            "top_n": int(args.top_n),

            "clr_eps": float(args.clr_eps),

        },

        "bootstrap": {"n_boot": int(args.n_boot)},

        "permutation": {"n_perm": int(args.n_perm)},

        "selected_best_loco": sel,

        "missing": missing,

    }

    (outdir / "panel_report.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] Saved:")

    print("  -", rep_path)

    print("  -", outdir / "panel_report.json")

    if missing:

        print("\n[WARN] Missing inputs:")

        for m in missing:

            print(" ", m)

    if sel is not None:

        print("\n[SELECT] Best (LOCO) by score:")

        print(f"  panel={sel.get('panel')} postprocess={sel.get('postprocess')} mean_cosine={sel.get('mean_cosine'):.4f} "

              f"sparsity_ratio={sel.get('sparsity_ratio'):.3f} voc_corr_median={sel.get('voc_corr_median'):.3f}")

    else:

        print("\n[INFO] No LOCO entries found to select best panel.")

if __name__ == "__main__":

    main()
