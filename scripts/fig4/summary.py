import pandas as pd

from pathlib import Path

from scipy.stats import spearmanr

out_dir = Path("results/paper_tables")

out_dir.mkdir(parents=True, exist_ok=True)

runs = {

    "LOCO_S1S4": Path("results/pred_vectors/run_20260131_234102_S1_S4/ideotype_v3/ideotype_ranking_v3.weighted_cosine.f1rho.thresh.fpPen0.20.absolute_log1p.rhof1.csv"),

    "LOCO_S4": Path("results/pred_vectors/run_20260131_234113_S4/ideotype_v3/ideotype_ranking_v3.weighted_cosine.f1rho.thresh.fpPen0.20.absolute_log1p.rhof1.csv"),

    "FITALL_S1S4": Path("results/pred_vectors/run_20260131_234106_S1_S4_FITALL/ideotype_v3/ideotype_ranking_v3.weighted_cosine.f1rho.thresh.fpPen0.20.absolute_log1p.rhof1.csv"),

    "FITALL_S4": Path("results/pred_vectors/run_20260131_234117_S4_FITALL/ideotype_v3/ideotype_ranking_v3.weighted_cosine.f1rho.thresh.fpPen0.20.absolute_log1p.rhof1.csv"),

}

rows = []

for name, p in runs.items():

    df = pd.read_csv(p)

    score_col = "pred_sim_pen" if "pred_sim_pen" in df.columns else "pred_sim"

    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    top5 = df.head(5)["cultivar"].tolist()

    df_ex = df[df["cultivar"] != "MTH"].copy()

    top3_ex = df_ex.head(3)["cultivar"].tolist()

    top5_ex = df_ex.head(5)["cultivar"].tolist()

    rho = spearmanr(df_ex["pred_sim_pen"], df_ex["true_sim"]).correlation

    rows.append({

        "run": name,

        "score_col": score_col,

        "spearman": rho,

        "top5_include_anchor": ",".join(top5),

        "top3_exclude_anchor": ",".join(top3_ex),

        "top5_exclude_anchor": ",".join(top5_ex),

    })

out_main = pd.DataFrame(rows)

out_main_path = out_dir / "main_table_penalty_fpPen0.20.ideotype_topk.csv"

out_main.to_csv(out_main_path, index=False)

print("\n[OK] wrote main summary:", out_main_path)

out_supplement = pd.DataFrame(rows)

out_supplement_path = out_dir / "supplement_table_penalty_fpPen0.20.ideotype_topk.csv"

out_supplement.to_csv(out_supplement_path, index=False)

print("\n[OK] wrote supplementary summary:", out_supplement_path)
