import numpy as np

import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]

MID  = ROOT / "intermediate"

OUT  = ROOT / "results"

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["pdf.fonttype"] = 42

plt.rcParams["ps.fonttype"] = 42

plt.rcParams["svg.fonttype"] = "none"

plt.rcParams["font.family"] = "Arial"

STAGE_ORDER = ["S1","S2","S3","S4"]

def compute_ranks(X, Y):

    S = cosine_similarity(X, Y)

    n = S.shape[0]

    ranks = np.empty(n, dtype=int)

    for i in range(n):

        order = np.argsort(-S[i])

        ranks[i] = int(np.where(order == i)[0][0]) + 1

    return ranks

def topk(r, k): return float((r <= k).mean())

def mrr(r): return float((1.0 / r).mean())

def null_perm(X, Y, obs, n_perm=200, seed=1):

    rng = np.random.default_rng(seed)

    n = X.shape[0]

    null = {"top1":[], "top5":[], "top10":[], "mrr":[]}

    for _ in range(n_perm):

        perm = rng.permutation(n)

        rr = compute_ranks(X, Y[perm])

        null["top1"].append(topk(rr, 1))

        null["top5"].append(topk(rr, 5))

        null["top10"].append(topk(rr, 10))

        null["mrr"].append(mrr(rr))

    out = {}

    for k, arr in null.items():

        arr = np.array(arr)

        out[f"null_{k}_mean"] = float(arr.mean())

        out[f"null_{k}_lo"] = float(np.quantile(arr, 0.025))

        out[f"null_{k}_hi"] = float(np.quantile(arr, 0.975))

        out[f"p_{k}"] = float((arr >= obs[k]).mean())

    return out

def eval_task(leaf_df, peel_df, tasks, n_perm=200, seed=1):

    leaf_df = leaf_df.sort_values("key").reset_index(drop=True)

    peel_df = peel_df.sort_values("key").reset_index(drop=True)

    assert (leaf_df["key"].values == peel_df["key"].values).all(), "Keys not aligned."

    X = leaf_df[tasks].to_numpy(dtype=float)

    Y = peel_df[tasks].to_numpy(dtype=float)

    rr = compute_ranks(X, Y)

    obs = dict(

        n=len(rr),

        top1=topk(rr,1),

        top5=topk(rr,5),

        top10=topk(rr,10),

        mrr=mrr(rr),

        base_top1=1.0/len(rr),

        base_top5=5.0/len(rr),

        base_top10=10.0/len(rr),

    )

    null = null_perm(X, Y, obs, n_perm=n_perm, seed=seed)

    return rr, {**obs, **null}

def draw(ax, met, title):

    xs = np.arange(3)

    labels = ["Top-1","Top-5","Top-10"]

    obs = [met["top1"], met["top5"], met["top10"]]

    base = [met["base_top1"], met["base_top5"], met["base_top10"]]

    null_mu = [met["null_top1_mean"], met["null_top5_mean"], met["null_top10_mean"]]

    null_lo = [met["null_top1_lo"], met["null_top5_lo"], met["null_top10_lo"]]

    null_hi = [met["null_top1_hi"], met["null_top5_hi"], met["null_top10_hi"]]

    ax.bar(xs-0.18, obs, width=0.36, label="Observed (relative)")

    ax.bar(xs+0.18, base, width=0.20, label="Random baseline")

    yerr = [np.array(null_mu)-np.array(null_lo), np.array(null_hi)-np.array(null_mu)]

    ax.errorbar(xs, null_mu, yerr=yerr, fmt='o', capsize=3, label="Null (perm) mean±95%")

    for i, v in enumerate(obs):

        ax.text(xs[i]-0.18, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(xs, labels)

    ax.set_ylim(0, min(1.0, max(max(obs), max(null_hi))*1.25 + 1e-6))

    ax.set_title(title)

    ax.set_ylabel("Retrieval accuracy")

    pv = f"p(top1)={met['p_top1']:.3g}, p(top5)={met['p_top5']:.3g}, p(top10)={met['p_top10']:.3g}"

    ax.text(0.02, 0.02, pv, transform=ax.transAxes, fontsize=9, va="bottom")

def main(n_perm=200, seed=1):

    tasks = Path(MID / "tasks.txt").read_text(encoding="utf-8").splitlines()

    leaf = pd.read_csv(MID / "sample_desc_leaf_relative_v4.csv")

    peel = pd.read_csv(MID / "sample_desc_peel_relative_v4.csv")

    leaf1 = leaf.copy(); peel1 = peel.copy()

    leaf1["key"] = leaf1["PairID"].astype(str)

    peel1["key"] = peel1["PairID"].astype(str)

    rr1, met1 = eval_task(leaf1, peel1, tasks, n_perm=n_perm, seed=seed)

    leaf2 = leaf.groupby(["Cultivar","Stage"], as_index=False)[tasks].mean()

    peel2 = peel.groupby(["Cultivar","Stage"], as_index=False)[tasks].mean()

    leaf2["key"] = leaf2["Cultivar"].astype(str) + "|" + leaf2["Stage"].astype(str)

    peel2["key"] = peel2["Cultivar"].astype(str) + "|" + peel2["Stage"].astype(str)

    rr2, met2 = eval_task(leaf2, peel2, tasks, n_perm=n_perm, seed=seed)

    pd.DataFrame([{"task":"sample_pair", **met1}, {"task":"cultivar_stage_centroid", **met2}]).to_csv(

        OUT / "Fig6D_metrics_relative_main_v4.csv", index=False

    )

    pd.DataFrame({"rank": rr1}).to_csv(OUT / "Fig6D_ranks_sample_pair_relative_v4.csv", index=False)

    pd.DataFrame({"rank": rr2}).to_csv(OUT / "Fig6D_ranks_centroid_relative_v4.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(13.0, 5.0))

    draw(axs[0], met1, "Sample-level (PairID) retrieval")

    draw(axs[1], met2, "Cultivar×Stage centroid retrieval")

    axs[0].legend(frameon=False, loc="upper right")

    axs[1].legend(frameon=False, loc="upper right")

    fig.suptitle("Fig6D(v4) | Cross-organ semantic retrieval (relative-weighted)")

    fig.tight_layout(rect=[0,0,1,0.95])

    fig.savefig(OUT / "Fig6D_retrieval_with_null_relative_main_v4.pdf")

    fig.savefig(OUT / "Fig6D_retrieval_with_null_relative_main_v4.svg")

    print("[OK]", OUT / "Fig6D_retrieval_with_null_relative_main_v4.pdf")

    fig2, ax2 = plt.subplots(1, 2, figsize=(13.0, 4.6))

    ax2[0].hist(rr1, bins=30)

    ax2[0].set_title("Rank distribution | Sample-level")

    ax2[0].set_xlabel("Rank of true paired Peel"); ax2[0].set_ylabel("Count")

    ax2[1].hist(rr2, bins=20)

    ax2[1].set_title("Rank distribution | Centroid-level")

    ax2[1].set_xlabel("Rank of matched Peel centroid"); ax2[1].set_ylabel("Count")

    fig2.tight_layout()

    fig2.savefig(OUT / "Fig6D_rank_distributions_relative_main_v4.pdf")

    fig2.savefig(OUT / "Fig6D_rank_distributions_relative_main_v4.svg")

    print("[OK]", OUT / "Fig6D_rank_distributions_relative_main_v4.pdf")

    leaf["Stage"] = pd.Categorical(leaf["Stage"], categories=STAGE_ORDER, ordered=True)

    peel["Stage"] = pd.Categorical(peel["Stage"], categories=STAGE_ORDER, ordered=True)

    stage_rows = []

    for st in STAGE_ORDER:

        Ls = leaf[leaf["Stage"].astype(str).eq(st)].groupby("Cultivar", as_index=False)[tasks].mean()

        Ps = peel[peel["Stage"].astype(str).eq(st)].groupby("Cultivar", as_index=False)[tasks].mean()

        if len(Ls)==0 or len(Ps)==0:

            continue

        Ls["key"]=Ls["Cultivar"].astype(str); Ps["key"]=Ps["Cultivar"].astype(str)

        rr, met = eval_task(Ls, Ps, tasks, n_perm=n_perm, seed=seed)

        stage_rows.append({"Stage": st, **met})

    pd.DataFrame(stage_rows).to_csv(OUT / "Fig6D_by_stage_centroid_metrics_relative_main_v4.csv", index=False)

    print("[OK]", OUT / "Fig6D_by_stage_centroid_metrics_relative_main_v4.csv")

if __name__ == "__main__":

    main()
