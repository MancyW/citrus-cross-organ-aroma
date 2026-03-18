import numpy as np

import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"

MID  = ROOT / "intermediate"

OUT  = ROOT / "results"

MID.mkdir(parents=True, exist_ok=True)

OUT.mkdir(parents=True, exist_ok=True)

STAGE_ORDER = ["S1","S2","S3","S4"]

def parse_pairid(pid: str):

    c, s, b, r = str(pid).split("_")

    return c, s, b, int(r)

def reduce_2d(X, seed=1):

    try:

        import umap

        return umap.UMAP(

            n_components=2, random_state=seed, n_neighbors=25, min_dist=0.15

        ).fit_transform(X)

    except Exception:

        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(X)

def main(seed=1):

    X = pd.read_csv(DATA / "X_leaf.csv")

    Y = pd.read_csv(DATA / "Y_peel.csv")

    assert X.shape == Y.shape, "X_leaf 与 Y_peel 形状不一致"

    assert (X["PairID"].astype(str).values == Y["PairID"].astype(str).values).all(), "PairID 不一一对应"

    voc_cols = [c for c in X.columns if c != "PairID"]

    leaf_sum = X[voc_cols].sum(axis=1).to_numpy()

    peel_sum = Y[voc_cols].sum(axis=1).to_numpy()

    qc = pd.DataFrame({

        "PairID": X["PairID"].astype(str),

        "Cultivar": X["PairID"].astype(str).str.split("_").str[0],

        "Stage": X["PairID"].astype(str).str.split("_").str[1],

        "Batch": X["PairID"].astype(str).str.split("_").str[2],

        "Rep": X["PairID"].astype(str).str.split("_").str[3],

        "leaf_sum": leaf_sum,

        "peel_sum": peel_sum,

        "flag_peel_sum_zero": (peel_sum == 0),

        "flag_leaf_sum_zero": (leaf_sum == 0),

    })

    qc.to_csv(OUT / "Fig6_QC_sum_by_sample_v4.csv", index=False)

    drop_peel0 = qc.loc[qc["flag_peel_sum_zero"], "PairID"].tolist()

    (OUT / "Fig6_QC_peel_sum0_pairids_v4.txt").write_text("\n".join(drop_peel0), encoding="utf-8")

    print(f"[QC] total samples: {len(qc)}")

    print(f"[QC] peel_sum==0 samples: {len(drop_peel0)} (will be filtered)")

    print(f"[QC] leaf_sum==0 samples: {int(qc['flag_leaf_sum_zero'].sum())} (reported only)")

    keep_mask = ~(qc["flag_peel_sum_zero"].to_numpy())

    Xf = X.loc[keep_mask].reset_index(drop=True)

    Yf = Y.loc[keep_mask].reset_index(drop=True)

    Xf.to_csv(MID / "X_leaf.filtered_peel0_removed_v4.csv", index=False)

    Yf.to_csv(MID / "Y_peel.filtered_peel0_removed_v4.csv", index=False)

    P = np.load(MID / "voc_pom_pred.npy")

    tasks = Path(MID / "tasks.txt").read_text(encoding="utf-8").splitlines()

    voc_dict_path = MID / "voc_dict.tsv"

    if voc_dict_path.exists():

        voc_dict = pd.read_csv(voc_dict_path, sep="\t")

        voc_order = voc_dict["VOC"].tolist()

        if voc_order != voc_cols:

            idx_map = {v:i for i,v in enumerate(voc_order)}

            try:

                idx = [idx_map[v] for v in voc_cols]

            except KeyError as e:

                raise RuntimeError(f"VOC name {e} not found in voc_dict.tsv; cannot align P.") from None

            P = P[idx, :]

    assert P.shape[0] == len(voc_cols), f"P rows ({P.shape[0]}) != n_voc ({len(voc_cols)})"

    assert P.shape[1] == len(tasks), f"P cols ({P.shape[1]}) != n_tasks ({len(tasks)})"

    def build_S(df_abund):

        A = df_abund[voc_cols].to_numpy(dtype=float)

        A = np.clip(A, 0, None)

        denom = A.sum(axis=1, keepdims=True) + 1e-12

        W = A / denom

        S = W @ P

        return S

    S_leaf = build_S(Xf)

    S_peel = build_S(Yf)

    def make_meta(pairids, organ):

        meta = pd.DataFrame({"PairID": [str(p) for p in pairids]})

        meta[["Cultivar","Stage","Batch","Rep"]] = meta["PairID"].str.split("_", expand=True)

        meta["Organ"] = organ

        meta["Stage"] = pd.Categorical(meta["Stage"], categories=STAGE_ORDER, ordered=True)

        return meta

    m_leaf = make_meta(Xf["PairID"].tolist(), "Leaf")

    m_peel = make_meta(Yf["PairID"].tolist(), "Peel")

    leaf_df = pd.concat([m_leaf, pd.DataFrame(S_leaf, columns=tasks)], axis=1)

    peel_df = pd.concat([m_peel, pd.DataFrame(S_peel, columns=tasks)], axis=1)

    leaf_df.to_csv(MID / "sample_desc_leaf_relative_v4.csv", index=False)

    peel_df.to_csv(MID / "sample_desc_peel_relative_v4.csv", index=False)

    S_all = np.vstack([S_leaf, S_peel])

    xy = reduce_2d(S_all, seed=seed)

    samp2d = pd.concat([

        m_leaf.assign(x=xy[:len(m_leaf),0], y=xy[:len(m_leaf),1]),

        m_peel.assign(x=xy[len(m_leaf):,0], y=xy[len(m_leaf):,1]),

    ], ignore_index=True)

    samp2d.to_csv(MID / "sample_desc_2d_relative_v4.csv", index=False)

    print("[OK] wrote filtered X/Y to:", MID / "X_leaf.filtered_peel0_removed_v4.csv")

    print("[OK] wrote relative sample desc to:", MID / "sample_desc_leaf_relative_v4.csv")

    print("[OK] wrote 2D coords to:", MID / "sample_desc_2d_relative_v4.csv")

    print("[OK] QC tables to:", OUT / "Fig6_QC_sum_by_sample_v4.csv")

if __name__ == "__main__":

    main()
