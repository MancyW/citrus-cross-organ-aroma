import argparse

from pathlib import Path

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from scipy.stats import spearmanr

def ensure_dir(p: Path):

    p.mkdir(parents=True, exist_ok=True)

def load_weights(weights_csv: Path) -> dict[str, float]:

    df = pd.read_csv(weights_csv)

    voc_col = None

    for c in ["VOC", "voc", "feature", "metabolite"]:

        if c in df.columns:

            voc_col = c

            break

    if voc_col is None:

        raise ValueError(f"Cannot find VOC column in {weights_csv}. cols={list(df.columns)}")

    w_col = None

    for c in ["weight", "w", "coef", "beta", "weight_aligned", "weight_trainonly"]:

        if c in df.columns:

            w_col = c

            break

    if w_col is None:

        num_cols = [c for c in df.columns if c != voc_col and pd.api.types.is_numeric_dtype(df[c])]

        if len(num_cols) == 1:

            w_col = num_cols[0]

        else:

            raise ValueError(f"Cannot find weight column in {weights_csv}. cols={list(df.columns)}")

    w = df[[voc_col, w_col]].dropna()

    return dict(zip(w[voc_col].astype(str), w[w_col].astype(float)))

def spearman(a, b) -> float:

    r = spearmanr(a, b).correlation

    return float(r) if np.isfinite(r) else np.nan

def bootstrap_delta(y_true, ai_pred, base_pred, B=20000, seed=0):

    rng = np.random.default_rng(seed)

    n = len(y_true)

    r_ai = []

    r_base = []

    d = []

    idx = np.arange(n)

    for _ in range(B):

        s = rng.choice(idx, size=n, replace=True)

        ra = spearman(y_true[s], ai_pred[s])

        rb = spearman(y_true[s], base_pred[s])

        r_ai.append(ra)

        r_base.append(rb)

        d.append(ra - rb)

    r_ai = np.array(r_ai, float)

    r_base = np.array(r_base, float)

    d = np.array(d, float)

    p = float(np.nanmean(d <= 0))

    return {

        "ai_mean": float(np.nanmean(r_ai)),

        "base_mean": float(np.nanmean(r_base)),

        "delta_mean": float(np.nanmean(d)),

        "delta_ci025": float(np.nanquantile(d, 0.025)),

        "delta_ci975": float(np.nanquantile(d, 0.975)),

        "p_value": p,

    }

def ai_peel_index_from_pred_vectors(pred_vectors_long: Path, weights: dict[str, float], y_stage: str):

    df = pd.read_parquet(pred_vectors_long)

    def pick(cols):

        for c in cols:

            if c in df.columns:

                return c

        return None

    col_cult = pick(["cultivar", "Cultivar", "cultivar_name"])

    col_voc = pick(["voc", "VOC", "feature"])

    col_true = pick(["y_true", "true", "obs", "y"])

    col_pred = pick(["y_pred", "pred", "yhat"])

    col_stage = pick(["stage", "Stage"])

    col_organ = pick(["organ", "Organ"])

    if col_cult is None or col_voc is None or col_true is None or col_pred is None:

        raise ValueError(f"pred_vectors_long schema not recognized. cols={list(df.columns)[:50]}")

    if col_stage is not None:

        df = df[df[col_stage].astype(str) == str(y_stage)]

    if col_organ is not None:

        df = df[df[col_organ].astype(str).str.lower() == "peel"]

    df[col_voc] = df[col_voc].astype(str)

    df = df[df[col_voc].isin(weights)].copy()

    if df.empty:

        raise ValueError("No VOC overlap between pred_vectors_long and weights.")

    df["w"] = df[col_voc].map(weights).astype(float)

    df["true_contrib"] = df[col_true].astype(float) * df["w"]

    df["pred_contrib"] = df[col_pred].astype(float) * df["w"]

    col_rep = pick(["sample_id", "sample", "replicate", "row_id", "obs_id"])

    gcols = [col_cult] + ([col_rep] if col_rep is not None else [])

    idx = df.groupby(gcols, dropna=False, as_index=False)[["true_contrib", "pred_contrib"]].sum()

    agg = idx.groupby(col_cult, as_index=False)[["true_contrib", "pred_contrib"]].mean()

    y_true = agg["true_contrib"].to_numpy(float)

    y_pred = agg["pred_contrib"].to_numpy(float)

    cultivars = agg[col_cult].astype(str).tolist()

    return cultivars, y_true, y_pred

def baseline_rf_from_clean_ssot(repo_root: Path, weights: dict[str, float], x_stages: list[str], y_stage: str, seed: int):

    p = repo_root / "data" / "ssot" / "ssot_cultivar_stage.clean.parquet"

    if not p.exists():

        raise FileNotFoundError(f"Missing clean SSOT: {p}")

    ss = pd.read_parquet(p)

    meta_cols = set(["cultivar", "Cultivar", "cultivar_name", "organ", "Organ", "stage", "Stage"])

    col_cult = "cultivar" if "cultivar" in ss.columns else ("Cultivar" if "Cultivar" in ss.columns else None)

    col_stage = "stage" if "stage" in ss.columns else ("Stage" if "Stage" in ss.columns else None)

    col_organ = "organ" if "organ" in ss.columns else ("Organ" if "Organ" in ss.columns else None)

    if col_cult is None or col_stage is None or col_organ is None:

        raise ValueError(f"clean SSOT schema not recognized. cols={list(ss.columns)[:30]}")

    voc_cols = [c for c in ss.columns if c not in meta_cols]

    leaf = ss[(ss[col_organ].astype(str).str.lower() == "leaf") & (ss[col_stage].astype(str).isin(x_stages))].copy()

    peel = ss[(ss[col_organ].astype(str).str.lower() == "peel") & (ss[col_stage].astype(str) == str(y_stage))].copy()

    leaf_long = leaf[[col_cult, col_stage] + voc_cols].copy()

    leaf_long[voc_cols] = np.log1p(leaf_long[voc_cols].astype(float))

    leaf_long["stage_prefix"] = leaf_long[col_stage].astype(str) + "::"

    X_blocks = []

    for st in x_stages:

        sub = leaf_long[leaf_long[col_stage].astype(str) == st].copy()

        if sub.empty:

            continue

        sub = sub.set_index(col_cult)[voc_cols]

        sub.columns = [f"{st}::{c}" for c in sub.columns]

        X_blocks.append(sub)

    if not X_blocks:

        raise ValueError("No leaf data found for x_stages in clean SSOT.")

    X = pd.concat(X_blocks, axis=1).fillna(0.0)

    peel_mat = peel.set_index(col_cult)[voc_cols].astype(float)

    peel_mat = np.log1p(peel_mat)

    common = [v for v in peel_mat.columns if v in weights]

    if not common:

        raise ValueError("No VOC overlap between clean SSOT peel VOCs and weights.")

    w = np.array([weights[v] for v in common], float)

    y = (peel_mat[common].to_numpy(float) @ w)

    cultivars = sorted(set(X.index.astype(str)) & set(peel_mat.index.astype(str)))

    X = X.loc[cultivars]

    y = pd.Series(y, index=peel_mat.index.astype(str)).loc[cultivars].to_numpy(float)

    return cultivars, X.to_numpy(float), y

def loco_predict_rf(cultivars, X, y, seed: int):

    n = len(cultivars)

    preds = np.full(n, np.nan, float)

    for i in range(n):

        train_idx = np.array([j for j in range(n) if j != i])

        test_idx = i

        model = RandomForestRegressor(

            n_estimators=500,

            random_state=seed,

            n_jobs=-1

        )

        model.fit(X[train_idx], y[train_idx])

        preds[test_idx] = model.predict(X[[test_idx]])[0]

    return preds

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--outdir", required=True)

    ap.add_argument("--pred_vectors_long", required=True)

    ap.add_argument("--weights_csv", required=True)

    ap.add_argument("--x_stages", nargs="*", default=["S1", "S4"])

    ap.add_argument("--y_stage", default="S4")

    ap.add_argument("--B", type=int, default=20000)

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[2]

    ensure_dir(outdir / "extras")

    weights = load_weights(Path(args.weights_csv))

    cult_ai, y_true_ai, y_pred_ai = ai_peel_index_from_pred_vectors(

        Path(args.pred_vectors_long), weights, args.y_stage

    )

    cult_base, X, y = baseline_rf_from_clean_ssot(repo_root, weights, args.x_stages, args.y_stage, args.seed)

    pred_base = loco_predict_rf(cult_base, X, y, args.seed)

    ix_ai = {c: i for i, c in enumerate(cult_ai)}

    ix_b = {c: i for i, c in enumerate(cult_base)}

    common = [c for c in cult_base if c in ix_ai]

    if len(common) < 5:

        raise ValueError(f"Too few common cultivars between AI and baseline: {len(common)}")

    y_true = np.array([y[ix_b[c]] for c in common], float)

    ai_pred = np.array([y_pred_ai[ix_ai[c]] for c in common], float)

    base_pred = np.array([pred_base[ix_b[c]] for c in common], float)

    r_ai = spearman(y_true, ai_pred)

    r_base = spearman(y_true, base_pred)

    stats = bootstrap_delta(y_true, ai_pred, base_pred, B=args.B, seed=args.seed)

    out_csv = outdir / "extras" / "ai_vs_bestbaseline_bootstrap.csv"

    out_md = outdir / "extras" / "ai_vs_bestbaseline_bootstrap.md"

    out_txt = outdir / "extras" / "ai_vs_bestbaseline_bootstrap.summary.txt"

    df = pd.DataFrame([{

        "n_cultivars": len(common),

        "AI_spearman": r_ai,

        "baseline": "RandomForest(cleanSSOT)",

        "baseline_spearman": r_base,

        "delta_mean": stats["delta_mean"],

        "delta_ci025": stats["delta_ci025"],

        "delta_ci975": stats["delta_ci975"],

        "p_value": stats["p_value"],

        "B": args.B,

        "seed": args.seed,

    }])

    df.to_csv(out_csv, index=False)

    with out_md.open("w", encoding="utf-8") as f:

        f.write("# AI vs best baseline (paired bootstrap, LOCO)\n\n")

        f.write(df.to_markdown(index=False) + "\n")

    with out_txt.open("w", encoding="utf-8") as f:

        f.write(

            f"n_cultivars={len(common)}\n"

            f"AI_spearman={r_ai:.6f}\n"

            f"baseline=RandomForest(cleanSSOT)\n"

            f"baseline_spearman={r_base:.6f}\n"

            f"delta_mean={stats['delta_mean']:.6f}\n"

            f"delta_CI95=[{stats['delta_ci025']:.6f}, {stats['delta_ci975']:.6f}]\n"

            f"p_value={stats['p_value']:.6f}\n"

            f"B={args.B}\nseed={args.seed}\n"

        )

    print(f"[OK] wrote: {out_csv}")

    print(f"[OK] wrote: {out_md}")

    print(f"[OK] wrote: {out_txt}")

if __name__ == "__main__":

    main()
