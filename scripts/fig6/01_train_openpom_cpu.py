import os, re, inspect, urllib.request

import numpy as np

import pandas as pd

from pathlib import Path

os.environ["DGLBACKEND"] = "pytorch"

import deepchem as dc

from openpom.feat.graph_featurizer import GraphFeaturizer

from openpom.models.mpnn_pom import MPNNPOMModel

try:

    from openpom.utils.data_utils import get_class_imbalance_ratio

    HAS_IMB = True

except Exception:

    HAS_IMB = False

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"

MODEL_DIR = ROOT / "models" / "openpom_ckpt"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

CURATED_URL = "URL_PLACEHOLDER"

def guess_smiles_field(df: pd.DataFrame) -> str:

    for cand in ["nonStereoSMILES", "smiles", "SMILES", "canonical_smiles", "CanonicalSMILES"]:

        if cand in df.columns:

            return cand

    for c in df.columns:

        if "smiles" in c.lower():

            return c

    raise ValueError("Cannot find SMILES column in curated dataset.")

def select_numeric_tasks(df: pd.DataFrame, smiles_field: str, min_convert_rate=0.98):

    tasks = []

    out = df.copy()

    for c in df.columns:

        if c == smiles_field:

            continue

        s = pd.to_numeric(df[c], errors="coerce")

        rate = float(s.notna().mean())

        if rate >= min_convert_rate:

            out[c] = s.fillna(0.0).astype(float)

            tasks.append(c)

    return out, tasks

def build_multihot_from_text(df: pd.DataFrame, smiles_field: str):

    obj_cols = [c for c in df.columns if c != smiles_field and df[c].dtype == object]

    if not obj_cols:

        raise ValueError("No object/text column found to build multi-hot labels.")

    best = None

    best_score = -1

    n = len(df)

    for c in obj_cols:

        s = df[c].fillna("").astype(str)

        delim_rate = s.str.contains(r"[;,|]").mean()

        uniq = s.nunique(dropna=True)

        score = float(delim_rate) * 2.0 + (1.0 - min(1.0, uniq / max(1, n)))

        if score > best_score:

            best_score, best = score, c

    labels = df[best].fillna("").astype(str)

    def split_tokens(x: str):

        x = x.strip()

        if not x:

            return []

        if re.search(r"[;,|]", x):

            toks = re.split(r"[;,|]\s*", x)

        else:

            toks = [x]

        toks = [t.strip() for t in toks if t.strip()]

        return toks

    token_lists = labels.apply(split_tokens)

    vocab = sorted({t for lst in token_lists for t in lst if t})

    if len(vocab) < 10:

        raise ValueError(f"Text label column '{best}' yields too few labels ({len(vocab)}).")

    Y = np.zeros((len(df), len(vocab)), dtype=float)

    idx = {t:i for i,t in enumerate(vocab)}

    for i, lst in enumerate(token_lists):

        for t in lst:

            Y[i, idx[t]] = 1.0

    clean = pd.concat([df[[smiles_field]].copy(), pd.DataFrame(Y, columns=vocab)], axis=1)

    tasks = vocab

    return clean, tasks, best

def main(nb_epoch=8, seed=1):

    curated = DATA / "curated_GS_LF_merged_4983.csv"

    if not curated.exists():

        print(f"[INFO] downloading curated dataset from: {CURATED_URL}")

        urllib.request.urlretrieve(CURATED_URL, curated)

    df = pd.read_csv(curated)

    smiles_field = guess_smiles_field(df)

    print(f"[INFO] smiles_field = {smiles_field}")

    clean_df, tasks = select_numeric_tasks(df, smiles_field, min_convert_rate=0.98)

    label_source = "numeric_columns"

    if len(tasks) < 20:

        clean_df, tasks, text_col = build_multihot_from_text(df, smiles_field)

        label_source = f"multihot_from_text:{text_col}"

    clean_path = DATA / "curated_clean_for_openpom.csv"

    clean_df.to_csv(clean_path, index=False)

    (MODEL_DIR / "tasks.txt").write_text("\n".join(tasks), encoding="utf-8")

    print(f"[INFO] label_source = {label_source}")

    print(f"[INFO] n_tasks = {len(tasks)} | clean_csv = {clean_path}")

    print(f"[INFO] tasks saved to: {MODEL_DIR/'tasks.txt'}")

    featurizer = GraphFeaturizer()

    loader = dc.data.CSVLoader(tasks=tasks, feature_field=smiles_field, featurizer=featurizer)

    dataset = loader.create_dataset([str(clean_path)])

    splitter = dc.splits.RandomStratifiedSplitter()

    train, test, valid = splitter.train_valid_test_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed)

    class_imb = None

    if HAS_IMB:

        class_imb = get_class_imbalance_ratio(train)

    lr = dc.models.optimizers.ExponentialDecay(

        initial_rate=0.001, decay_rate=0.5, decay_steps=32 * 20, staircase=True

    )

    sig = inspect.signature(MPNNPOMModel.__init__)

    kwargs = dict(

        n_tasks=len(tasks),

        batch_size=128,

        learning_rate=lr,

        model_dir=str(MODEL_DIR),

        class_imbalance_ratio=class_imb,

    )

    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}

    model = MPNNPOMModel(**kwargs)

    print(f"[INFO] Training OpenPOM-style model on CPU | epochs={nb_epoch} | model_dir={MODEL_DIR}")

    model.fit(train, nb_epoch=nb_epoch)

    model.save_checkpoint(model_dir=str(MODEL_DIR))

    print(f"[OK] saved checkpoint to {MODEL_DIR}")

if __name__ == "__main__":

    main()
