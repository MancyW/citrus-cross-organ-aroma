from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

def spearman(x: np.ndarray, y: np.ndarray) -> float:

    rx = pd.Series(x).rank(method="average").to_numpy()

    ry = pd.Series(y).rank(method="average").to_numpy()

    rx = rx - rx.mean()

    ry = ry - ry.mean()

    denom = (np.linalg.norm(rx) * np.linalg.norm(ry)) + 1e-12

    return float(np.dot(rx, ry) / denom)

def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:

    na = np.linalg.norm(a) + eps

    nb = np.linalg.norm(b) + eps

    return float(np.dot(a, b) / (na * nb))

def to_composition(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:

    v = np.asarray(v, dtype=float)

    v = np.clip(v, 0.0, None)

    s = v.sum()

    if s <= 0:

        return np.full_like(v, 1.0 / len(v))

    p = v / s

    p = np.clip(p, eps, None)

    p = p / p.sum()

    return p

def clr(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:

    p = np.asarray(p, dtype=float)

    p = np.clip(p, eps, None)

    lp = np.log(p)

    return lp - lp.mean()

def jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:

    p = np.clip(p, eps, None); p = p / p.sum()

    q = np.clip(q, eps, None); q = q / q.sum()

    m = 0.5 * (p + q)

    def kl(a, b):

        return float(np.sum(a * (np.log(a) - np.log(b))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def similarity(u: np.ndarray, v: np.ndarray, metric: str) -> float:

    if metric == "cosine":

        return cosine(u, v)

    if metric == "jsd_sim":

        d = jsd(u, v)

        return float(1.0 - np.sqrt(max(d, 0.0)))

    raise ValueError(f"Unknown metric: {metric}")

def build_ideotype(true_w: pd.DataFrame, mode: str, anchor: str | None, top_n: int) -> np.ndarray:

    M = true_w.to_numpy(dtype=float)

    if mode == "cultivar":

        if anchor is None or anchor not in true_w.index:

            raise ValueError(f"anchor '{anchor}' not in cultivars: {true_w.index.tolist()}")

        return true_w.loc[anchor].to_numpy(dtype=float)

    totals = np.log1p(M).sum(axis=1)

    if mode == "mean_top_total":

        idx = np.argsort(-totals)[:max(1, int(top_n))]

        return M[idx].mean(axis=0)

    if mode == "mean_all":

        return M.mean(axis=0)

    raise ValueError(f"Unknown ideotype mode: {mode}")

def topk_recall(df: pd.DataFrame, k: int) -> float:

    k = max(1, min(int(k), len(df)))

    top_pred = df.sort_values("pred_sim", ascending=False).head(k)["cultivar"].tolist()

    top_true = df.sort_values("true_sim", ascending=False).head(k)["cultivar"].tolist()

    return len(set(top_pred).intersection(set(top_true))) / float(k)

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--pred_long", required=True, help="pred_vectors_long.parquet OR .csv")

    ap.add_argument("--outdir", default=None)

    ap.add_argument("--ideotype_mode", default="cultivar", choices=["cultivar", "mean_top_total", "mean_all"])

    ap.add_argument("--anchor", default=None)

    ap.add_argument("--top_n", type=int, default=3)

    ap.add_argument("--space", default="composition", choices=["absolute_log1p", "composition", "clr"])

    ap.add_argument("--metric", default="cosine", choices=["cosine", "jsd_sim"])

    args = ap.parse_args()

    p = Path(args.pred_long)

    outdir = Path(args.outdir) if args.outdir else p.parent / "ideotype_v2"

    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)

    need = {"cultivar", "VOC", "y_pred", "y_true"}

    miss = need - set(df.columns)

    if miss:

        raise KeyError(f"Missing columns in {p}: {sorted(miss)}")

    true_w = df.pivot_table(index="cultivar", columns="VOC", values="y_true", aggfunc="first").fillna(0.0).sort_index()

    pred_w = df.pivot_table(index="cultivar", columns="VOC", values="y_pred", aggfunc="first").fillna(0.0).reindex(true_w.index)

    ide_abs = build_ideotype(true_w, args.ideotype_mode, args.anchor, args.top_n)

    rows = []

    for c in true_w.index:

        yt = true_w.loc[c].to_numpy(dtype=float)

        yp = pred_w.loc[c].to_numpy(dtype=float)

        if args.space == "absolute_log1p":

            ut = np.log1p(yt); up = np.log1p(yp); ui = np.log1p(ide_abs)

        elif args.space == "composition":

            ut = to_composition(yt); up = to_composition(yp); ui = to_composition(ide_abs)

        elif args.space == "clr":

            ut = clr(to_composition(yt)); up = clr(to_composition(yp)); ui = clr(to_composition(ide_abs))

        else:

            raise ValueError(args.space)

        rows.append({

            "cultivar": c,

            "true_sim": similarity(ut, ui, args.metric),

            "pred_sim": similarity(up, ui, args.metric),

            "true_total_logsum": float(np.log1p(yt).sum()),

            "pred_total_logsum": float(np.log1p(yp).sum()),

        })

    out = pd.DataFrame(rows).sort_values("pred_sim", ascending=False)

    summary = {

        "ideotype_mode": args.ideotype_mode,

        "anchor": args.anchor,

        "top_n": args.top_n,

        "space": args.space,

        "metric": args.metric,

        "spearman(pred_sim,true_sim)": spearman(out["pred_sim"].to_numpy(), out["true_sim"].to_numpy()),

        "top1_recall": topk_recall(out, 1),

        "top3_recall": topk_recall(out, 3),

        "top5_recall": topk_recall(out, 5),

    }

    out.to_csv(outdir / "ideotype_ranking_v2.csv", index=False)

    Path(outdir / "ideotype_summary_v2.txt").write_text(

        "\n".join([f"{k}: {v}" for k, v in summary.items()]),

        encoding="utf-8"

    )

    print("[OK] Saved:")

    print("  -", outdir / "ideotype_ranking_v2.csv")

    print("  -", outdir / "ideotype_summary_v2.txt")

    print("\n[Summary]")

    for k, v in summary.items():

        print(f"  {k}: {v}")

    print("\n[Preview] top 10 by predicted similarity:")

    print(out.head(10)[["cultivar","pred_sim","true_sim","true_total_logsum","pred_total_logsum"]].to_string(index=False))

if __name__ == "__main__":

    main()
