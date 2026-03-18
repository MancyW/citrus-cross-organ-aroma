import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]

MID  = ROOT / "intermediate"

OUT  = ROOT / "results"

OUT.mkdir(parents=True, exist_ok=True)

def top_items(vec, names, k=10):

    idx = np.argsort(-vec)[:k]

    return [(names[i], float(vec[i])) for i in idx]

def main(top_desc=10, top_voc=10):

    voc2d = pd.read_csv(MID / "voc_pom_2d.csv")

    P = np.load(MID / "voc_pom_pred.npy")

    tasks = Path(MID / "tasks.txt").read_text(encoding="utf-8").splitlines()

    assert len(voc2d) == P.shape[0]

    assert len(tasks) == P.shape[1]

    x = voc2d["x"].to_numpy()

    y = voc2d["y"].to_numpy()

    x0 = np.median(x); y0 = np.median(y)

    region_defs = {

        "Q1(+x,+y)": (x >= x0) & (y >= y0),

        "Q2(-x,+y)": (x <  x0) & (y >= y0),

        "Q3(-x,-y)": (x <  x0) & (y <  y0),

        "Q4(+x,-y)": (x >= x0) & (y <  y0),

    }

    rows_region = []

    rows_desc = []

    rows_anchor = []

    for rid, m in region_defs.items():

        n = int(m.sum())

        if n < 5:

            continue

        Pm = P[m]

        mu = Pm.mean(axis=0)

        desc = top_items(mu, tasks, k=top_desc)

        for rank, (d, score) in enumerate(desc, start=1):

            rows_desc.append({"region": rid, "rank": rank, "descriptor": d, "mean_prob": score})

        sim = cosine_similarity(P, mu.reshape(1, -1)).ravel()

        sim_in = sim.copy()

        sim_in[~m] = -1e9

        idx = np.argsort(-sim_in)[:top_voc]

        for rank, i in enumerate(idx, start=1):

            rows_anchor.append({

                "region": rid,

                "rank": rank,

                "VOC": voc2d.loc[i, "VOC"],

                "Family": voc2d.loc[i, "Family"],

                "Subclass": voc2d.loc[i, "Subclass"],

                "x": float(voc2d.loc[i, "x"]),

                "y": float(voc2d.loc[i, "y"]),

                "sim_to_region_mean": float(sim[i]),

            })

        rows_region.append({

            "region": rid,

            "n_voc": n,

            "x_median": float(np.median(x[m])),

            "y_median": float(np.median(y[m])),

        })

    df_region = pd.DataFrame(rows_region)

    df_desc = pd.DataFrame(rows_desc)

    df_anchor = pd.DataFrame(rows_anchor)

    df_region.to_csv(OUT / "Fig6A_region_summary_v3.csv", index=False)

    df_desc.to_csv(OUT / "Fig6A_region_top_descriptors_v3.csv", index=False)

    df_anchor.to_csv(OUT / "Fig6A_region_anchor_VOCs_v3.csv", index=False)

    compact = []

    for rid in df_region["region"].tolist():

        topD = df_desc[df_desc["region"].eq(rid)].sort_values("rank")["descriptor"].tolist()

        topV = df_anchor[df_anchor["region"].eq(rid)].sort_values("rank")["VOC"].tolist()

        compact.append({

            "region": rid,

            "n_voc": int(df_region[df_region["region"].eq(rid)]["n_voc"].iloc[0]),

            "top_descriptors": "; ".join(topD),

            "anchor_VOCs": "; ".join(topV),

        })

    pd.DataFrame(compact).to_csv(OUT / "Fig6A_region_support_table_v3.csv", index=False)

    print("[OK]", OUT / "Fig6A_region_support_table_v3.csv")

    print("[OK]", OUT / "Fig6A_region_top_descriptors_v3.csv")

    print("[OK]", OUT / "Fig6A_region_anchor_VOCs_v3.csv")

if __name__ == "__main__":

    main()
