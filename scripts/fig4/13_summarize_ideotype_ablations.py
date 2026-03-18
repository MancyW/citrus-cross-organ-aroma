import argparse

import re

from pathlib import Path

import numpy as np

import pandas as pd

from scipy.stats import spearmanr

PAT = re.compile(

    r"^ideotype_ranking_v3\.(?P<out_tag>.+?)\.fpPen(?P<fp>\d+\.\d+)\.(?P<space>[^.]+)\.(?P<wm>[^.]+)\.csv$"

)

def topk_lists(rank: pd.DataFrame, anchor: str, score_col: str, k: int):

    pred = rank.sort_values(score_col, ascending=False)["cultivar"].astype(str).tolist()

    true = rank.sort_values("true_sim", ascending=False)["cultivar"].astype(str).tolist()

    pred_top = pred[:k]

    true_top = true[:k]

    overlap = len(set(pred_top) & set(true_top))

    pred_ex = [c for c in pred if c != anchor][:k]

    true_ex = [c for c in true if c != anchor][:k]

    overlap_ex = len(set(pred_ex) & set(true_ex))

    return {

        f"top{k}_include_anchor_pred": ",".join(pred_top),

        f"top{k}_include_anchor_true": ",".join(true_top),

        f"top{k}_include_anchor_overlap": overlap,

        f"top{k}_exclude_anchor_pred": ",".join(pred_ex),

        f"top{k}_exclude_anchor_true": ",".join(true_ex),

        f"top{k}_exclude_anchor_overlap": overlap_ex,

    }

def safe_spearman(a, b):

    a = np.asarray(a, dtype=float)

    b = np.asarray(b, dtype=float)

    try:

        r = spearmanr(a, b).correlation

    except Exception:

        r = np.nan

    if r is None:

        r = np.nan

    return float(r)

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--anchor", default="MTH")

    ap.add_argument("--base_tag_prefix", default="weighted_cosine.f1rho.thresh",

                    help="Only summarize files whose out_tag starts with this prefix")

    ap.add_argument("--run_dirs", nargs="*", default=[

        "results/pred_vectors/run_20260131_234102_S1_S4",

        "results/pred_vectors/run_20260131_234113_S4",

        "results/pred_vectors/run_20260131_234106_S1_S4_FITALL",

        "results/pred_vectors/run_20260131_234117_S4_FITALL",

    ])

    ap.add_argument("--out_csv", default="results/paper_tables/ablation_ideotype_metrics.csv")

    args = ap.parse_args()

    out_path = Path(args.out_csv)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for rd in map(Path, args.run_dirs):

        ideod = rd / "ideotype_v3"

        if not ideod.exists():

            continue

        for f in sorted(ideod.glob("ideotype_ranking_v3.*.csv")):

            m = PAT.match(f.name)

            if not m:

                continue

            out_tag = m.group("out_tag")

            if not out_tag.startswith(args.base_tag_prefix):

                continue

            fp = float(m.group("fp"))

            space = m.group("space")

            wm = m.group("wm")

            df = pd.read_csv(f)

            if "cultivar" not in df.columns or "true_sim" not in df.columns:

                continue

            score_col = "pred_sim_pen" if "pred_sim_pen" in df.columns else "pred_sim"

            df["cultivar"] = df["cultivar"].astype(str)

            spearman_inc = safe_spearman(df[score_col], df["true_sim"])

            df_ex = df[df["cultivar"] != args.anchor].copy()

            spearman_ex = safe_spearman(df_ex[score_col], df_ex["true_sim"])

            rec = {

                "run_dir": str(rd),

                "run_name": rd.name,

                "ranking_csv": str(f),

                "out_tag": out_tag,

                "fp_penalty_lambda": fp,

                "space": space,

                "weight_mode": wm,

                "score_col": score_col,

                "spearman_include_anchor": spearman_inc,

                "spearman_exclude_anchor": spearman_ex,

            }

            for k in [1, 3, 5]:

                rec.update(topk_lists(df, args.anchor, score_col, k))

            rows.append(rec)

    if not rows:

        raise RuntimeError("No matching ablation ranking CSV found. Did you run Step 3.1?")

    out = pd.DataFrame(rows)

    out["calibrate"] = out["out_tag"].apply(lambda s: str(s).endswith(".calib"))

    out = out.sort_values(

        ["run_name", "space", "weight_mode", "calibrate", "fp_penalty_lambda"],

        ascending=[True, True, True, True, True],

    )

    out.to_csv(out_path, index=False)

    print("[OK] wrote:", out_path)

    print("\n[Preview] best per run_name (by spearman_exclude_anchor):")

    for rn, g in out.groupby("run_name"):

        g2 = g.sort_values("spearman_exclude_anchor", ascending=False).head(5)

        cols = ["run_name","space","weight_mode","calibrate","fp_penalty_lambda",

                "spearman_exclude_anchor","top3_exclude_anchor_pred","top5_exclude_anchor_pred"]

        print("\n---", rn, "---")

        print(g2[cols].to_string(index=False))

if __name__ == "__main__":

    main()
