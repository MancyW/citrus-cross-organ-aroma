import numpy as np

import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]

MID  = ROOT / "intermediate"

OUT  = ROOT / "results"

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["pdf.fonttype"] = 42

plt.rcParams["ps.fonttype"] = 42

plt.rcParams["svg.fonttype"] = "none"

plt.rcParams["font.family"] = "Arial"

def build_weights(A, scheme, topm=None):

    A = np.clip(A, 0, None).astype(float)

    if topm is not None:

        idx = np.argpartition(-A, kth=min(topm, A.shape[1]-1), axis=1)[:, :topm]

        mask = np.zeros_like(A, dtype=bool)

        row = np.arange(A.shape[0])[:, None]

        mask[row, idx] = True

        A = np.where(mask, A, 0.0)

    if scheme == "relative":

        W = A

    elif scheme == "log1p":

        W = np.log1p(A)

    elif scheme == "sqrt":

        W = np.sqrt(A)

    else:

        raise ValueError("scheme must be relative/log1p/sqrt")

    W = W / (W.sum(axis=1, keepdims=True) + 1e-12)

    return W

def ranks(X, Y):

    S = cosine_similarity(X, Y)

    n = S.shape[0]

    rr = np.empty(n, dtype=int)

    for i in range(n):

        order = np.argsort(-S[i])

        rr[i] = int(np.where(order == i)[0][0]) + 1

    return rr

def metrics(rr):

    rr = np.asarray(rr, dtype=float)

    return dict(

        top1=float((rr <= 1).mean()),

        top5=float((rr <= 5).mean()),

        top10=float((rr <= 10).mean()),

        mrr=float((1.0/rr).mean()),

        n=int(len(rr)),

    )

def bootstrap_ci(rr, B=500, seed=1):

    rng = np.random.default_rng(seed)

    rr = np.asarray(rr, dtype=int)

    n = len(rr)

    stats = {"top1":[], "top5":[], "top10":[], "mrr":[]}

    for _ in range(B):

        idx = rng.integers(0, n, size=n)

        m = metrics(rr[idx])

        for k in stats:

            stats[k].append(m[k])

    out = {}

    for k, v in stats.items():

        v = np.array(v)

        out[k+"_lo"] = float(np.quantile(v, 0.025))

        out[k+"_hi"] = float(np.quantile(v, 0.975))

    return out

def main(B=500, seed=1):

    P = np.load(MID / "voc_pom_pred.npy")

    tasks = Path(MID / "tasks.txt").read_text(encoding="utf-8").splitlines()

    Xf = pd.read_csv(MID / "X_leaf.filtered_peel0_removed_v4.csv")

    Yf = pd.read_csv(MID / "Y_peel.filtered_peel0_removed_v4.csv")

    voc_cols = [c for c in Xf.columns if c != "PairID"]

    A_leaf = Xf[voc_cols].to_numpy()

    A_peel = Yf[voc_cols].to_numpy()

    meta = pd.DataFrame({"PairID": Xf["PairID"].astype(str)})

    meta[["Cultivar","Stage","Batch","Rep"]] = meta["PairID"].str.split("_", expand=True)

    meta["key"] = meta["Cultivar"].astype(str) + "|" + meta["Stage"].astype(str)

    scheme_specs = [

        ("relative", "relative", None),

        ("log1p", "log1p", None),

        ("sqrt", "sqrt", None),

        ("relative_top20", "relative", 20),

        ("relative_top50", "relative", 50),

    ]

    rows = []

    for name, scheme, topm in scheme_specs:

        Wl = build_weights(A_leaf, scheme=scheme, topm=topm)

        Wp = build_weights(A_peel, scheme=scheme, topm=topm)

        Sl = Wl @ P

        Sp = Wp @ P

        rr1 = ranks(Sl, Sp)

        m1 = metrics(rr1); ci1 = bootstrap_ci(rr1, B=B, seed=seed)

        rows.append({"scheme": name, "task":"sample_pair", **m1, **ci1})

        leaf_cent = pd.DataFrame(Sl, columns=tasks).assign(key=meta["key"]).groupby("key", as_index=False)[tasks].mean().sort_values("key")

        peel_cent = pd.DataFrame(Sp, columns=tasks).assign(key=meta["key"]).groupby("key", as_index=False)[tasks].mean().sort_values("key")

        rr2 = ranks(leaf_cent[tasks].to_numpy(), peel_cent[tasks].to_numpy())

        m2 = metrics(rr2); ci2 = bootstrap_ci(rr2, B=min(B,400), seed=seed)

        rows.append({"scheme": name, "task":"cultivar_stage_centroid", **m2, **ci2})

    df = pd.DataFrame(rows)

    df.to_csv(OUT / "Fig6E_robustness_metrics_after_qc_v4.csv", index=False)

    def plot_panel(ax, task):

        sub = df[df["task"].eq(task)].sort_values("scheme")

        x = np.arange(len(sub))

        y = sub["top10"].to_numpy()

        ylo = sub["top10_lo"].to_numpy()

        yhi = sub["top10_hi"].to_numpy()

        ax.bar(x, y)

        ax.errorbar(x, y, yerr=[y-ylo, yhi-y], fmt="none", capsize=3, lw=1)

        for i, v in enumerate(y):

            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x, sub["scheme"].tolist(), rotation=25, ha="right")

        ax.set_ylabel("Top-10 (95% bootstrap CI)")

        ax.set_title(task)

    fig, axs = plt.subplots(2, 1, figsize=(10.6, 8.8))

    plot_panel(axs[0], "sample_pair")

    plot_panel(axs[1], "cultivar_stage_centroid")

    fig.suptitle("Fig6E(v4) | Robustness across weighting schemes after QC filtering")

    fig.tight_layout(rect=[0,0,1,0.96])

    fig.savefig(OUT / "Fig6E_weighting_robustness_after_qc_v4.pdf")

    fig.savefig(OUT / "Fig6E_weighting_robustness_after_qc_v4.svg")

    print("[OK]", OUT / "Fig6E_weighting_robustness_after_qc_v4.pdf")

if __name__ == "__main__":

    main()
