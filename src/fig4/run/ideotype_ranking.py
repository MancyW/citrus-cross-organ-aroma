from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np

import pandas as pd

def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:

    a = np.asarray(a, dtype=float)

    b = np.asarray(b, dtype=float)

    na = np.linalg.norm(a) + eps

    nb = np.linalg.norm(b) + eps

    return float(np.dot(a, b) / (na * nb))

def build_ideotype(vecs_true: pd.DataFrame, mode: str, anchor: str | None, top_n: int) -> np.ndarray:

    M = vecs_true.to_numpy(dtype=float)

    if mode == "cultivar":

        if anchor is None or anchor not in vecs_true.index:

            raise ValueError(f"anchor cultivar '{anchor}' not found in cultivars: {vecs_true.index.tolist()}")

        return vecs_true.loc[anchor].to_numpy(dtype=float)

    totals = np.log1p(M).sum(axis=1)

    if mode == "mean_top_total":

        idx = np.argsort(-totals)[:max(1, int(top_n))]

        return M[idx].mean(axis=0)

    if mode == "mean_all":

        return M.mean(axis=0)

    raise ValueError(f"Unknown ideotype mode: {mode}")

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--pred_long", required=True, help="pred_vectors_long.parquet OR pred_vectors_long.csv")

    ap.add_argument("--outdir", default=None)

    ap.add_argument("--ideotype_mode", default="cultivar", choices=["cultivar", "mean_top_total", "mean_all"])

    ap.add_argument("--anchor", default=None, help="cultivar code if ideotype_mode=cultivar, e.g. MTH")

    ap.add_argument("--top_n", type=int, default=3, help="used if ideotype_mode=mean_top_total")

    ap.add_argument("--use_log1p", action="store_true", help="compute cosine in log1p space")

    args = ap.parse_args()

    pred_path = Path(args.pred_long)

    outdir = Path(args.outdir) if args.outdir else pred_path.parent / "ideotype"

    outdir.mkdir(parents=True, exist_ok=True)

    if pred_path.suffix.lower() == ".parquet":

        df = pd.read_parquet(pred_path)

    else:

        df = pd.read_csv(pred_path)

    need = {"cultivar", "VOC", "y_pred", "y_true"}

    miss = need - set(df.columns)

    if miss:

        raise KeyError(f"Missing columns in {pred_path}: {sorted(miss)}")

    true_w = df.pivot_table(index="cultivar", columns="VOC", values="y_true", aggfunc="first").fillna(0.0)

    pred_w = df.pivot_table(index="cultivar", columns="VOC", values="y_pred", aggfunc="first").fillna(0.0)

    true_w = true_w.sort_index()

    pred_w = pred_w.reindex(true_w.index)

    ideotype_true = build_ideotype(true_w, args.ideotype_mode, args.anchor, args.top_n)

    rows = []

    for c in true_w.index:

        yt = true_w.loc[c].to_numpy(dtype=float)

        yp = pred_w.loc[c].to_numpy(dtype=float)

        if args.use_log1p:

            yt_eval = np.log1p(yt)

            yp_eval = np.log1p(yp)

            ide_eval = np.log1p(ideotype_true)

        else:

            yt_eval, yp_eval, ide_eval = yt, yp, ideotype_true

        rows.append({

            "cultivar": c,

            "true_cos_to_ideotype": cosine(yt_eval, ide_eval),

            "pred_cos_to_ideotype": cosine(yp_eval, ide_eval),

            "true_total_logsum": float(np.log1p(yt).sum()),

            "pred_total_logsum": float(np.log1p(yp).sum()),

        })

    out = pd.DataFrame(rows).sort_values("pred_cos_to_ideotype", ascending=False)

    def topk_recall(k: int) -> float:

        k = max(1, min(int(k), len(out)))

        top_pred = out.sort_values("pred_cos_to_ideotype", ascending=False).head(k)["cultivar"].tolist()

        top_true = out.sort_values("true_cos_to_ideotype", ascending=False).head(k)["cultivar"].tolist()

        return len(set(top_pred).intersection(set(top_true))) / float(k)

    ks = [1, 3, 5]

    summary = {

        "ideotype_mode": args.ideotype_mode,

        "anchor": args.anchor,

        "top_n": args.top_n,

        "use_log1p": bool(args.use_log1p),

        "top1_recall": topk_recall(1),

        "top3_recall": topk_recall(3),

        "top5_recall": topk_recall(5),

    }

    out.to_csv(outdir / "ideotype_ranking.csv", index=False)

    Path(outdir / "ideotype_summary.txt").write_text(

        "\n".join([f"{k}: {v}" for k, v in summary.items()]),

        encoding="utf-8"

    )

    print("[OK] Saved:")

    print("  -", outdir / "ideotype_ranking.csv")

    print("  -", outdir / "ideotype_summary.txt")

    print("\n[Preview] top 10 by predicted cosine-to-ideotype:")

    print(out.head(10)[["cultivar","pred_cos_to_ideotype","true_cos_to_ideotype","true_total_logsum","pred_total_logsum"]].to_string(index=False))

    print("\n[Recall] top1/top3/top5:", summary["top1_recall"], summary["top3_recall"], summary["top5_recall"])

if __name__ == "__main__":

    main()
