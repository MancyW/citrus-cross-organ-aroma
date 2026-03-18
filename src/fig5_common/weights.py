from __future__ import annotations

import pandas as pd

from pathlib import Path

def load_weights(weights_csv: str) -> pd.DataFrame:

    df = pd.read_csv(weights_csv)

    cols = {c: c.strip() for c in df.columns}

    df = df.rename(columns=cols)

    voc_col = None

    for c in ["VOC", "voc", "feature", "Feature", "name", "Name"]:

        if c in df.columns:

            voc_col = c; break

    if voc_col is None:

        raise ValueError(f"Cannot find VOC column in weights_csv: {weights_csv}")

    cand = [c for c in df.columns if c.lower() in ["weight", "w", "coef", "beta"]]

    if not cand:

        num = df.drop(columns=[voc_col]).select_dtypes(include="number").columns.tolist()

        if not num:

            raise ValueError("Cannot infer weight column (no numeric cols).")

        wcol = num[0]

    else:

        wcol = cand[0]

    out = df[[voc_col, wcol]].rename(columns={voc_col: "VOC", wcol: "weight"})

    out["VOC"] = out["VOC"].astype(str)

    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")

    out = out.dropna(subset=["VOC"])

    return out

def topN_vocs(w: pd.DataFrame, N: int = 30) -> list[str]:

    ww = w.dropna(subset=["weight"]).copy()

    ww["absw"] = ww["weight"].abs()

    ww = ww.sort_values("absw", ascending=False).head(int(N))

    return ww["VOC"].tolist()

def infer_default_weights(repo_root: str, run_id: str) -> str:

    p = Path(repo_root) / "results" / "pred_vectors" / run_id / "ideotype_v3"

    hard = p / "voc_weights.weighted_cosine.f1rho.thresh.fpPen0.20.absolute_log1p.rhof1.csv"

    if hard.exists():

        return str(hard)

    cands = sorted(p.glob("voc_weights*.csv"))

    if not cands:

        raise FileNotFoundError(f"No voc_weights*.csv found under {p}")

    return str(cands[0])
