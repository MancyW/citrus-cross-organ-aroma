import argparse

from pathlib import Path

import numpy as np

import pandas as pd

from scipy.stats import spearmanr

def ensure_dir(p: Path):

    p.mkdir(parents=True, exist_ok=True)

def load_weights(weights_csv: Path) -> dict[str, float]:

    df = pd.read_csv(weights_csv)

    voc_col = next((c for c in ["VOC", "voc", "feature"] if c in df.columns), None)

    if voc_col is None:

        raise ValueError(f"Cannot find VOC col in {weights_csv}")

    w_col = next((c for c in ["weight", "w", "coef", "beta"] if c in df.columns), None)

    if w_col is None:

        num_cols = [c for c in df.columns if c != voc_col and pd.api.types.is_numeric_dtype(df[c])]

        if len(num_cols) != 1:

            raise ValueError(f"Cannot infer weight col in {weights_csv}")

        w_col = num_cols[0]

    df = df[[voc_col, w_col]].dropna()

    return dict(zip(df[voc_col].astype(str), df[w_col].astype(float)))

def spearman(a, b) -> float:

    r = spearmanr(a, b).correlation

    return float(r) if np.isfinite(r) else np.nan

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--outdir", required=True)

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

    p = repo_root / "data" / "ssot" / "ssot_cultivar_stage.clean.parquet"

    if not p.exists():

        raise FileNotFoundError(f"Missing clean SSOT: {p}")

    ss = pd.read_parquet(p)

    col_cult = "cultivar" if "cultivar" in ss.columns else ("Cultivar" if "Cultivar" in ss.columns else None)

    col_stage = "stage" if "stage" in ss.columns else ("Stage" if "Stage" in ss.columns else None)

    col_organ = "organ" if "organ" in ss.columns else ("Organ" if "Organ" in ss.columns else None)

    if col_cult is None or col_stage is None or col_organ is None:

        raise ValueError("clean SSOT schema not recognized")

    meta = {col_cult, col_stage, col_organ}

    voc_cols = [c for c in ss.columns if c not in meta]

    peel = ss[(ss[col_organ].astype(str).str.lower() == "peel") & (ss[col_stage].astype(str) == str(args.y_stage))].copy()

    peel = peel.set_index(col_cult)[voc_cols].astype(float)

    peel = np.log1p(peel)

    common = [v for v in peel.columns if v in weights]

    if not common:

        raise ValueError("No VOC overlap between peel and weights.")

    w = np.array([weights[v] for v in common], float)

    y = (peel[common].to_numpy(float) @ w)

    cultivars = peel.index.astype(str).tolist()

    rng = np.random.default_rng(args.seed)

    null_rhos = []

    for _ in range(args.B):

        perm = rng.permutation(len(cultivars))

        null_rhos.append(spearman(y, y[perm]))

    null_rhos = np.array(null_rhos, float)

    out_csv = outdir / "extras" / "permutation_null.csv"

    out_md = outdir / "extras" / "permutation_null.md"

    df = pd.DataFrame({

        "rho_null": null_rhos

    })

    df.to_csv(out_csv, index=False)

    summary = pd.DataFrame([{

        "B": args.B,

        "seed": args.seed,

        "null_mean": float(np.nanmean(null_rhos)),

        "null_ci025": float(np.nanquantile(null_rhos, 0.025)),

        "null_ci975": float(np.nanquantile(null_rhos, 0.975)),

    }])

    with out_md.open("w", encoding="utf-8") as f:

        f.write("# Permutation null for Spearman (cultivar label break)\n\n")

        f.write(summary.to_markdown(index=False) + "\n\n")

        f.write(f"- wrote samples: `{out_csv}`\n")

    print(f"[OK] wrote: {out_csv}")

    print(f"[OK] wrote: {out_md}")

if __name__ == "__main__":

    main()
