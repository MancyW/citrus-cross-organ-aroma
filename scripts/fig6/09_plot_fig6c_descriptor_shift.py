import numpy as np

import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt

from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[2]

MID  = ROOT / "intermediate"

OUT  = ROOT / "results"

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["pdf.fonttype"] = 42

plt.rcParams["ps.fonttype"] = 42

plt.rcParams["svg.fonttype"] = "none"

plt.rcParams["font.family"] = "Arial"

STAGE_ORDER = ["S1","S2","S3","S4"]

def bh_fdr(pvals):

    p = np.asarray(pvals, dtype=float)

    n = p.size

    order = np.argsort(p)

    ranked = p[order]

    q = ranked * n / (np.arange(1, n+1))

    q = np.minimum.accumulate(q[::-1])[::-1]

    q = np.clip(q, 0, 1)

    out = np.empty_like(q)

    out[order] = q

    return out

def bootstrap_ci(x, B=2000, seed=1):

    rng = np.random.default_rng(seed)

    x = np.asarray(x, dtype=float)

    n = len(x)

    means = []

    for _ in range(B):

        idx = rng.integers(0, n, size=n)

        means.append(float(np.mean(x[idx])))

    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

def main(topk=15, B=2000, seed=1):

    tasks = Path(MID / "tasks.txt").read_text(encoding="utf-8").splitlines()

    leaf = pd.read_csv(MID / "sample_desc_leaf_relative_v4.csv")

    peel = pd.read_csv(MID / "sample_desc_peel_relative_v4.csv")

    leaf = leaf.sort_values("PairID").reset_index(drop=True)

    peel = peel.sort_values("PairID").reset_index(drop=True)

    assert (leaf["PairID"].astype(str).values == peel["PairID"].astype(str).values).all()

    D = peel[tasks].to_numpy(dtype=float) - leaf[tasks].to_numpy(dtype=float)

    n_pair = D.shape[0]

    means = D.mean(axis=0)

    pvals = []

    for t in tasks:

        _, p = ttest_rel(peel[t].to_numpy(), leaf[t].to_numpy(), nan_policy="omit")

        pvals.append(p)

    pvals = np.array(pvals, dtype=float)

    qvals = bh_fdr(pvals)

    ci_lo = np.empty(D.shape[1], dtype=float)

    ci_hi = np.empty(D.shape[1], dtype=float)

    for j in range(D.shape[1]):

        lo, hi = bootstrap_ci(D[:, j], B=B, seed=seed)

        ci_lo[j] = lo; ci_hi[j] = hi

    stat = pd.DataFrame({

        "descriptor": tasks,

        "mean_delta": means,

        "ci_lo": ci_lo,

        "ci_hi": ci_hi,

        "p_value": pvals,

        "q_value": qvals,

        "n_pair": n_pair,

        "weighting": "relative",

    })

    stat.to_csv(OUT / "Fig6C_descriptor_shift_stats_relative_v4.csv", index=False)

    sel = stat.reindex(stat["mean_delta"].abs().sort_values(ascending=False).index).head(topk).copy()

    sel = sel.sort_values("mean_delta")

    fig, ax = plt.subplots(figsize=(7.4, 5.8))

    y = np.arange(len(sel))

    ax.barh(y, sel["mean_delta"].to_numpy())

    ax.errorbar(

        sel["mean_delta"].to_numpy(), y,

        xerr=[sel["mean_delta"].to_numpy()-sel["ci_lo"].to_numpy(),

              sel["ci_hi"].to_numpy()-sel["mean_delta"].to_numpy()],

        fmt="none", capsize=3, lw=1

    )

    ax.axvline(0, lw=1)

    ax.set_yticks(y, sel["descriptor"].tolist())

    ax.set_xlabel("Mean(Peel − Leaf) in odor-descriptor probability (paired by tree; relative-weighted)")

    ax.set_title(f"Fig6C(v4) | Descriptor shift with 95% bootstrap CI (n={n_pair})")

    for i, (_, r) in enumerate(sel.iterrows()):

        if r["q_value"] < 0.05:

            xpos = r["ci_hi"] if r["mean_delta"] >= 0 else r["ci_lo"]

            ax.text(xpos, i, "  *", va="center", fontsize=12)

    fig.tight_layout()

    fig.savefig(OUT / "Fig6C_descriptor_shift_relative_v4.pdf")

    fig.savefig(OUT / "Fig6C_descriptor_shift_relative_v4.svg")

    print("[OK] Fig6C v4 written:", OUT / "Fig6C_descriptor_shift_relative_v4.pdf")

if __name__ == "__main__":

    main()
