import argparse

from pathlib import Path

import numpy as np

import pandas as pd

from scipy.stats import spearmanr

def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:

    a = np.asarray(a, dtype=float)

    b = np.asarray(b, dtype=float)

    if a.size == 0 or b.size == 0:

        return 0.0

    if np.all(a == a[0]) or np.all(b == b[0]):

        return 0.0

    r = spearmanr(a, b).correlation

    if r is None or np.isnan(r):

        return 0.0

    return float(r)

def ensure_dir(p: Path) -> None:

    p.mkdir(parents=True, exist_ok=True)

def qstr(q: float) -> str:

    return f"{q:.2f}"

def maybe_append_fp_pen_tag(out_tag: str, lam: float) -> str:

    if lam <= 0:

        return out_tag

    if "fpPen" in out_tag:

        return out_tag

    return out_tag + f".fpPen{lam:.2f}"

def build_thresholded_parquet(

    pv_path: Path,

    out_pv_path: Path,

    out_thr_csv: Path,

    q: float = 0.05,

    thresh_space: str = "log1p",

    min_pos: int = 1,

) -> None:

    pv = pd.read_parquet(pv_path).copy()

    for c in ["y_true", "y_pred"]:

        pv[c] = pv[c].astype(float)

    if thresh_space != "log1p":

        raise ValueError("Only thresh_space=log1p is supported currently.")

    vocs = pv["VOC"].unique().tolist()

    thr_rows = []

    grp = pv.groupby("VOC", sort=False)

    for voc, g in grp:

        tpos = g.loc[g["y_true"] > 0, "y_true"].astype(float).values

        npos = int((tpos > 0).sum())

        if npos < min_pos:

            thr_log = float("inf")

        else:

            thr_log = float(np.quantile(np.log1p(tpos), q))

        thr_rows.append((voc, thr_log, npos))

    thr = pd.DataFrame(thr_rows, columns=["VOC", "thr_log1p", "n_true_pos"]).set_index("VOC")

    ensure_dir(out_thr_csv.parent)

    thr.to_csv(out_thr_csv)

    pv = pv.join(thr, on="VOC")

    thr_log = pv["thr_log1p"].values

    yt_log = np.log1p(pv["y_true"].values)

    yp_log = np.log1p(pv["y_pred"].values)

    yt_keep = yt_log >= thr_log

    yp_keep = yp_log >= thr_log

    pv.loc[~yt_keep, "y_true"] = 0.0

    pv.loc[~yp_keep, "y_pred"] = 0.0

    pv = pv[["panel", "cultivar", "VOC", "y_pred", "y_true"]]

    ensure_dir(out_pv_path.parent)

    pv.to_parquet(out_pv_path, index=False)

def calibrate_predictions_log1p(

    pv: pd.DataFrame,

    min_points: int = 6,

) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = pv.copy()

    df["y_true"] = df["y_true"].astype(float)

    df["y_pred"] = df["y_pred"].astype(float)

    df["lt"] = np.log1p(df["y_true"].values)

    df["lp"] = np.log1p(df["y_pred"].values)

    rows = []

    for voc, g in df.groupby("VOC", sort=False):

        x = g["lp"].values.astype(float)

        y = g["lt"].values.astype(float)

        mask = ~(np.isnan(x) | np.isnan(y))

        x = x[mask]

        y = y[mask]

        n = int(len(x))

        if n < min_points or (np.all(x == x[0])):

            a, b = 0.0, 1.0

            used = 0

        else:

            X = np.column_stack([np.ones_like(x), x])

            beta, *_ = np.linalg.lstsq(X, y, rcond=None)

            a = float(beta[0])

            b = float(beta[1])

            used = n

        rows.append((voc, a, b, used))

    calib = pd.DataFrame(rows, columns=["VOC", "a_intercept", "b_slope", "n_points"]).set_index("VOC")

    df = df.join(calib, on="VOC")

    lp = np.log1p(df["y_pred"].values.astype(float))

    lp_cal = df["a_intercept"].values + df["b_slope"].values * lp

    lp_cal = np.maximum(lp_cal, 0.0)

    df["y_pred"] = np.expm1(lp_cal)

    df = df[["panel", "cultivar", "VOC", "y_pred", "y_true"]]

    return df, calib.reset_index()

def transform_space(mat: np.ndarray, space: str, eps: float = 1e-12) -> np.ndarray:

    X = np.asarray(mat, dtype=float)

    if space == "absolute_log1p":

        return X

    if space == "clr":

        y = np.expm1(np.maximum(X, 0.0))

        y = y + eps

        y = y / np.maximum(y.sum(axis=1, keepdims=True), eps)

        logy = np.log(y)

        return logy - logy.mean(axis=1, keepdims=True)

    if space == "zscore_log1p":

        mu = X.mean(axis=0, keepdims=True)

        sd = X.std(axis=0, keepdims=True)

        sd = np.where(sd < 1e-12, 1.0, sd)

        return (X - mu) / sd

    raise ValueError(f"Unknown space: {space}")

def wcos(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:

    u = np.asarray(u, dtype=float)

    v = np.asarray(v, dtype=float)

    w = np.asarray(w, dtype=float)

    sw = np.sqrt(np.maximum(w, 0.0))

    uw = u * sw

    vw = v * sw

    nu = np.linalg.norm(uw)

    nv = np.linalg.norm(vw)

    if nu <= 0 or nv <= 0:

        return 0.0

    return float(np.dot(uw, vw) / (nu * nv))

def wcos_contrib(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:

    u = np.asarray(u, dtype=float)

    v = np.asarray(v, dtype=float)

    w = np.asarray(w, dtype=float)

    sw = np.sqrt(np.maximum(w, 0.0))

    uw = u * sw

    vw = v * sw

    nu = np.linalg.norm(uw)

    nv = np.linalg.norm(vw)

    if nu <= 0 or nv <= 0:

        return np.zeros_like(u, dtype=float)

    denom = nu * nv

    return (w * u * v) / denom

def compute_voc_weights(pv: pd.DataFrame, vocs: list[str], weight_mode: str) -> pd.DataFrame:

    df = pv.copy()

    df["y_true"] = df["y_true"].astype(float)

    df["y_pred"] = df["y_pred"].astype(float)

    df["lt"] = np.log1p(df["y_true"].values)

    df["lp"] = np.log1p(df["y_pred"].values)

    rows = []

    for voc, g in df.groupby("VOC", sort=False):

        lt = g["lt"].values.astype(float)

        lp = g["lp"].values.astype(float)

        rho = safe_spearman(lp, lt)

        rho_pos = max(0.0, float(rho))

        true_pos = (g["y_true"].values.astype(float) > 0)

        pred_pos = (g["y_pred"].values.astype(float) > 0)

        tp = int(np.sum(true_pos & pred_pos))

        fp = int(np.sum((~true_pos) & pred_pos))

        fn = int(np.sum(true_pos & (~pred_pos)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        if weight_mode == "rho":

            w_raw = rho_pos

        elif weight_mode == "rhof1":

            w_raw = rho_pos * f1

        else:

            raise ValueError(f"Unknown weight_mode: {weight_mode}")

        rows.append((voc, rho_pos, f1, float(w_raw)))

    wdf = pd.DataFrame(rows, columns=["VOC", "rho_pos", "f1", "w_raw"]).set_index("VOC")

    wdf = wdf.reindex(vocs).fillna(0.0)

    mean_w = float(wdf["w_raw"].mean()) if len(wdf) else 0.0

    wdf["w_used"] = wdf["w_raw"] / (mean_w + 1e-12)

    return wdf

def load_pv(run_dir: Path, use_thresh: bool, pv_name: str, q: float, thresh_space: str, min_pos: int) -> tuple[pd.DataFrame, Path]:

    pv_path = run_dir / pv_name

    if not pv_path.exists():

        raise FileNotFoundError(f"Cannot find: {pv_path}")

    if not use_thresh:

        pv = pd.read_parquet(pv_path)

        return pv, pv_path

    out_pv = run_dir / f"pred_vectors_long.thresh_q{qstr(q)}_{thresh_space}.parquet"

    out_thr_csv = run_dir / "ideotype_v3" / f"voc_thresholds.q{qstr(q)}.{thresh_space}.minpos{min_pos}.csv"

    if not out_pv.exists():

        print(f"PV(thresh) missing, building from: {pv_path}")

        build_thresholded_parquet(

            pv_path=pv_path,

            out_pv_path=out_pv,

            out_thr_csv=out_thr_csv,

            q=q,

            thresh_space=thresh_space,

            min_pos=min_pos,

        )

        print(f"[OK] wrote: {out_pv}")

        print(f"[OK] wrote: {out_thr_csv}")

    pv = pd.read_parquet(out_pv)

    return pv, out_pv

def make_matrices(pv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:

    pv2 = pv.copy()

    pv2["y_true"] = pv2["y_true"].astype(float)

    pv2["y_pred"] = pv2["y_pred"].astype(float)

    true_mat = pv2.pivot(index="cultivar", columns="VOC", values="y_true").fillna(0.0)

    pred_mat = pv2.pivot(index="cultivar", columns="VOC", values="y_pred").fillna(0.0)

    vocs = sorted(list(set(true_mat.columns).union(set(pred_mat.columns))))

    true_mat = true_mat.reindex(columns=vocs, fill_value=0.0)

    pred_mat = pred_mat.reindex(columns=vocs, fill_value=0.0)

    cultivars = true_mat.index.tolist()

    return true_mat, pred_mat, cultivars, vocs

def compute_fp_rate(pv: pd.DataFrame) -> pd.Series:

    df = pv.copy()

    yt = df["y_true"].astype(float).values

    yp = df["y_pred"].astype(float).values

    is_tn = (yt == 0)

    is_fp = is_tn & (yp > 0)

    tmp = pd.DataFrame({

        "cultivar": df["cultivar"].values,

        "tn": is_tn.astype(int),

        "fp": is_fp.astype(int),

    })

    agg = tmp.groupby("cultivar", sort=False)[["tn", "fp"]].sum()

    rate = agg["fp"] / agg["tn"].clip(lower=1)

    return rate

def eval_ranking(rank: pd.DataFrame, anchor: str, score_col: str, ks=(1, 3, 5)) -> None:

    rho_in = safe_spearman(rank[score_col].values, rank["true_sim"].values)

    print(f"\n[include anchor] spearman = {rho_in:.6f} (n={len(rank)})")

    for k in ks:

        t = set(rank.sort_values("true_sim", ascending=False).head(k)["cultivar"].tolist())

        p = set(rank.sort_values(score_col, ascending=False).head(k)["cultivar"].tolist())

        inter = len(t & p)

        print(f"  top{k}: overlap={inter}/{k}  true={sorted(t)}  pred={sorted(p)}")

    rank2 = rank[rank["cultivar"] != anchor].copy()

    rho_ex = safe_spearman(rank2[score_col].values, rank2["true_sim"].values)

    print(f"[exclude anchor={anchor}] spearman = {rho_ex:.6f} (n={len(rank2)})")

    for k in ks:

        t = set(rank2.sort_values("true_sim", ascending=False).head(k)["cultivar"].tolist())

        p = set(rank2.sort_values(score_col, ascending=False).head(k)["cultivar"].tolist())

        inter = len(t & p)

        print(f"  top{k}: overlap={inter}/{k}  true={sorted(t)}  pred={sorted(p)}")

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--run_dir", required=True, type=str)

    ap.add_argument("--anchor", default="MTH", type=str)

    ap.add_argument("--use_thresh", action="store_true")

    ap.add_argument("--pv_name", default="pred_vectors_long.parquet", type=str)

    ap.add_argument("--out_tag", default="weighted_cosine", type=str)

    ap.add_argument("--space", default="absolute_log1p", choices=["absolute_log1p", "clr", "zscore_log1p"])

    ap.add_argument("--weight_mode", default="rhof1", choices=["rhof1", "rho"])

    ap.add_argument("--calibrate", action="store_true")

    ap.add_argument("--calib_min_points", default=6, type=int)

    ap.add_argument("--print_infl_weights", action="store_true")

    ap.add_argument("--fp_target", default=None, type=str)

    ap.add_argument("--contrib_target", default=None, type=str)

    ap.add_argument("--thresh_q", default=0.05, type=float)

    ap.add_argument("--thresh_space", default="log1p", type=str)

    ap.add_argument("--thresh_min_pos", default=1, type=int)

    ap.add_argument("--fp_penalty_lambda", default=0.0, type=float)

    args = ap.parse_args()

    run_dir = Path(args.run_dir)

    ideotype_dir = run_dir / "ideotype_v3"

    ensure_dir(ideotype_dir)

    print("RUN:", run_dir)

    print("PV :", (run_dir / args.pv_name) if not args.use_thresh else (run_dir / f"pred_vectors_long.thresh_q{qstr(args.thresh_q)}_{args.thresh_space}.parquet"))

    print(f"space: {args.space} | weight_mode: {args.weight_mode} | calibrate: {bool(args.calibrate)} | fp_penalty_lambda: {args.fp_penalty_lambda}")

    pv, pv_used_path = load_pv(

        run_dir=run_dir,

        use_thresh=args.use_thresh,

        pv_name=args.pv_name,

        q=args.thresh_q,

        thresh_space=args.thresh_space,

        min_pos=args.thresh_min_pos,

    )

    if args.calibrate:

        pv_cal, calib_df = calibrate_predictions_log1p(pv, min_points=args.calib_min_points)

        calib_path = ideotype_dir / f"voc_calibration.{args.out_tag}.csv"

        calib_df.to_csv(calib_path, index=False)

        out_cal_pv = run_dir / f"pred_vectors_long.calibrated.{args.out_tag}.parquet"

        pv_cal.to_parquet(out_cal_pv, index=False)

        print(f"[OK] wrote: {calib_path}")

        print(f"[OK] wrote: {out_cal_pv}")

        pv = pv_cal

    true_mat, pred_mat, cultivars, vocs = make_matrices(pv)

    if args.anchor not in cultivars:

        raise ValueError(f"anchor {args.anchor} not in cultivars: {cultivars}")

    X_true0 = np.log1p(true_mat.values.astype(float))

    X_pred0 = np.log1p(pred_mat.values.astype(float))

    X_true = transform_space(X_true0, args.space)

    X_pred = transform_space(X_pred0, args.space)

    wdf = compute_voc_weights(pv, vocs, args.weight_mode)

    weights = wdf["w_used"].values.astype(float)

    eff_tag = maybe_append_fp_pen_tag(args.out_tag, args.fp_penalty_lambda)

    w_out = ideotype_dir / f"voc_weights.{eff_tag}.{args.space}.{args.weight_mode}.csv"

    wdf.reset_index().to_csv(w_out, index=False)

    iA = cultivars.index(args.anchor)

    a = X_true[iA]

    rows = []

    for i, c in enumerate(cultivars):

        true_sim = wcos(X_true[i], a, weights)

        pred_sim = wcos(X_pred[i], a, weights)

        rows.append({

            "cultivar": c,

            "true_sim": float(true_sim),

            "pred_sim": float(pred_sim),

            "true_total_logsum": float(np.sum(X_true0[i])),

            "pred_total_logsum": float(np.sum(X_pred0[i])),

        })

    rank = pd.DataFrame(rows)

    score_col = "pred_sim"

    if args.fp_penalty_lambda and args.fp_penalty_lambda > 0:

        fp_rate = compute_fp_rate(pv)

        rank = rank.merge(fp_rate.rename("fp_rate"), on="cultivar", how="left").fillna({"fp_rate": 0.0})

        rank["pred_sim_pen"] = rank["pred_sim"] - float(args.fp_penalty_lambda) * rank["fp_rate"]

        score_col = "pred_sim_pen"

    rank = rank.sort_values(score_col, ascending=False).reset_index(drop=True)

    r_out = ideotype_dir / f"ideotype_ranking_v3.{eff_tag}.{args.space}.{args.weight_mode}.csv"

    rank.to_csv(r_out, index=False)

    print(f"[OK] wrote: {w_out}")

    print(f"[OK] wrote: {r_out}")

    eval_ranking(rank, anchor=args.anchor, score_col=score_col)

    if args.fp_target is not None:

        sub = pv[pv["cultivar"] == args.fp_target].copy()

        sub["y_true"] = sub["y_true"].astype(float)

        sub["y_pred"] = sub["y_pred"].astype(float)

        sub = sub[(sub["y_true"] == 0) & (sub["y_pred"] > 0)].copy()

        sub["logpred"] = np.log1p(sub["y_pred"].values)

        sub = sub.sort_values("logpred", ascending=False)

        print(f"\n[{args.fp_target}] FP VOCs (true=0, pred>0) top 30:")

        if len(sub) == 0:

            print("(none)")

        else:

            print(sub.head(30)[["VOC", "y_true", "y_pred", "logpred"]].to_string(index=False))

    if args.contrib_target is not None:

        tgt = args.contrib_target

        if tgt not in cultivars:

            print(f"\n[WARN] contrib_target={tgt} not in cultivars, skip.")

        else:

            iT = cultivars.index(tgt)

            contrib_true = wcos_contrib(X_true[iT], a, weights)

            contrib_pred = wcos_contrib(X_pred[iT], a, weights)

            delta = contrib_pred - contrib_true

            dbg = pd.DataFrame({

                "VOC": vocs,

                "delta_contrib(pred-true)": delta.astype(float),

                "y_true": true_mat.loc[tgt, vocs].values.astype(float),

                "y_pred": pred_mat.loc[tgt, vocs].values.astype(float),

            }).sort_values("delta_contrib(pred-true)", ascending=False)

            dbg_path = ideotype_dir / f"debug_delta_contrib.{tgt}_vs_{args.anchor}.{eff_tag}.{args.space}.{args.weight_mode}.csv"

            dbg.to_csv(dbg_path, index=False)

            print(f"\n[OK] wrote: {dbg_path}")

            print(f"\nTop 30 VOC that inflate pred_sim for {tgt} vs {args.anchor}:")

            print(dbg.head(30)[["VOC", "delta_contrib(pred-true)", "y_true", "y_pred"]].to_string(index=False))

            if args.print_infl_weights:

                topv = dbg.head(15)["VOC"].tolist()

                tmp = wdf.loc[topv, ["rho_pos", "f1", "w_used"]].copy()

                tmp.index.name = "VOC"

                print(f"\n[weights for top inflator VOCs] (rho_pos, f1, w_used)")

                print(tmp.to_string())

    if args.print_infl_weights and args.contrib_target is None:

        tmp = wdf.sort_values("w_used", ascending=False).head(20)[["rho_pos", "f1", "w_used"]]

        print("\n[top VOC weights] (rho_pos, f1, w_used) top 20:")

        print(tmp.to_string())

if __name__ == "__main__":

    main()
