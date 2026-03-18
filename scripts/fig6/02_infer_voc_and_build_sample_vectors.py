import os

import numpy as np

import pandas as pd

from pathlib import Path

os.environ["DGLBACKEND"] = "pytorch"

import deepchem as dc

from openpom.feat.graph_featurizer import GraphFeaturizer

from openpom.models.mpnn_pom import MPNNPOMModel

from sklearn.decomposition import PCA

try:

    import umap

    HAS_UMAP = True

except Exception:

    HAS_UMAP = False

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"

MID  = ROOT / "intermediate"

MODEL_DIR = ROOT / "models" / "openpom_ckpt"

MID.mkdir(parents=True, exist_ok=True)

STAGE_ORDER = ["S1","S2","S3","S4"]

def parse_pairid(pid: str):

    c, s, batch, rep = pid.split("_")

    return c, s, batch, int(rep)

def reduce_2d(X, seed=1):

    if HAS_UMAP:

        return umap.UMAP(n_components=2, random_state=seed, n_neighbors=25, min_dist=0.15).fit_transform(X)

    return PCA(n_components=2, random_state=seed).fit_transform(X)

def main(seed=1):

    tasks_path = MODEL_DIR / "tasks.txt"

    if not tasks_path.exists():

        raise FileNotFoundError(f"Missing {tasks_path}. Run scripts/01_train_openpom_cpu.py first.")

    tasks = tasks_path.read_text(encoding="utf-8").splitlines()

    (MID / "tasks.txt").write_text("\n".join(tasks), encoding="utf-8")

    voc = pd.read_csv(MID / "voc_dict.tsv", sep="\t")

    voc_cols = voc["VOC"].tolist()

    import inspect

    sig = inspect.signature(MPNNPOMModel.__init__)

    kwargs = dict(n_tasks=len(tasks), batch_size=128, model_dir=str(MODEL_DIR))

    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    model = MPNNPOMModel(**kwargs)

    model.restore(model_dir=str(MODEL_DIR))

    tmp = MID / "tmp_voc_for_infer.csv"

    tmp_df = pd.DataFrame({"nonStereoSMILES": voc["SMILES_canon"].tolist()})

    for t in tasks:

        tmp_df[t] = 0.0

    tmp_df.to_csv(tmp, index=False)

    featurizer = GraphFeaturizer()

    loader = dc.data.CSVLoader(tasks=tasks, feature_field="nonStereoSMILES", featurizer=featurizer)

    ds = loader.create_dataset([str(tmp)])

    P = model.predict(ds)

    E = P

    np.save(MID / "voc_pom_pred.npy", P)

    np.save(MID / "voc_pom_embed.npy", E)

    voc_xy = reduce_2d(E, seed=seed)

    voc_2d = voc[["VOC","Family","Subclass"]].copy()

    voc_2d["x"] = voc_xy[:,0]

    voc_2d["y"] = voc_xy[:,1]

    voc_2d.to_csv(MID / "voc_pom_2d.csv", index=False)

    x_leaf = pd.read_csv(DATA / "X_leaf.csv")

    y_peel = pd.read_csv(DATA / "Y_peel.csv")

    def build_sample_desc(df_abund):

        A = df_abund[voc_cols].to_numpy(dtype=float)

        W = np.log1p(np.clip(A, 0, None))

        W = W / (W.sum(axis=1, keepdims=True) + 1e-12)

        return W @ P

    S_leaf = build_sample_desc(x_leaf)

    S_peel = build_sample_desc(y_peel)

    def make_meta(pairids, organ):

        meta = pd.DataFrame({"PairID": pairids})

        meta[["Cultivar","Stage","Batch","Rep"]] = meta["PairID"].apply(lambda s: pd.Series(parse_pairid(s)))

        meta["Organ"] = organ

        meta["Stage"] = pd.Categorical(meta["Stage"], categories=STAGE_ORDER, ordered=True)

        return meta

    m_leaf = make_meta(x_leaf["PairID"].tolist(), "Leaf")

    m_peel = make_meta(y_peel["PairID"].tolist(), "Peel")

    leaf_df = pd.concat([m_leaf, pd.DataFrame(S_leaf, columns=tasks)], axis=1)

    peel_df = pd.concat([m_peel, pd.DataFrame(S_peel, columns=tasks)], axis=1)

    leaf_df.to_csv(MID / "sample_desc_leaf.csv", index=False)

    peel_df.to_csv(MID / "sample_desc_peel.csv", index=False)

    S_all = np.vstack([S_leaf, S_peel])

    samp_xy = reduce_2d(S_all, seed=seed)

    samp2d = pd.concat([

        m_leaf.assign(x=samp_xy[:len(m_leaf),0], y=samp_xy[:len(m_leaf),1]),

        m_peel.assign(x=samp_xy[len(m_leaf):,0], y=samp_xy[len(m_leaf):,1]),

    ], ignore_index=True)

    samp2d.to_csv(MID / "sample_desc_2d.csv", index=False)

    print("[OK] wrote voc/sample intermediates to", MID)

if __name__ == "__main__":

    main()
