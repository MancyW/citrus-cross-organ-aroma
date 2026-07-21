import pandas as pd

from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"

MID  = ROOT / "intermediate"

MID.mkdir(parents=True, exist_ok=True)

FAMILY_FIX = {

    "β-Myrcene": ("Monoterpenes", "Monoterpene hydrocarbons"),

}

def canon_smiles(s):

    m = Chem.MolFromSmiles(s)

    return None if m is None else Chem.MolToSmiles(m, canonical=True)

def main():

    x = pd.read_csv(DATA / "X_leaf.csv")

    y = pd.read_csv(DATA / "Y_peel.csv")

    assert x.shape == y.shape, "X_leaf 与 Y_peel 形状不一致"

    assert (x["PairID"].values == y["PairID"].values).all(), "PairID 不一一对应"

    voc_cols = x.columns[1:].tolist()

    fam = pd.read_csv(DATA / "voc_family.csv")

    smi = pd.read_csv(DATA / "voc_smiles.csv")

    voc = smi.merge(fam, on="VOC", how="left")

    for k, (famv, subc) in FAMILY_FIX.items():

        m = voc["VOC"].eq(k)

        voc.loc[m, "Family"] = voc.loc[m, "Family"].fillna(famv)

        voc.loc[m, "Subclass"] = voc.loc[m, "Subclass"].fillna(subc)

    voc["SMILES_canon"] = voc["SMILES"].apply(canon_smiles)

    voc = voc.set_index("VOC").loc[voc_cols].reset_index()

    voc.to_csv(MID / "voc_dict.tsv", sep="\t", index=False)

    print(f"[OK] wrote {MID/'voc_dict.tsv'} | n_voc={len(voc_cols)}")

if __name__ == "__main__":

    main()
