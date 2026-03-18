from __future__ import annotations

import re

from pathlib import Path

import numpy as np

import pandas as pd

META_BLACKLIST_DEFAULT = ["Cultivar", "Organ", "Stage", "Batch", "Rep"]

def read_table(path: str) -> pd.DataFrame:

    p = str(path)

    if p.endswith(".parquet"):

        return pd.read_parquet(p)

    return pd.read_csv(p)

def _infer_sample_col(df: pd.DataFrame) -> str:

    for c in ["SampleID", "sample_id", "sample", "ID", "id"]:

        if c in df.columns:

            return c

    return df.columns[0]

def standardize_sampleid(s: str) -> dict:

    s = str(s).strip()

    toks = s.split("_")

    cultivar = toks[0] if toks else s

    stage = None

    nums = []

    for t in toks[1:]:

        if re.match(r"^S\d+$", t, flags=re.IGNORECASE):

            stage = t.upper()

        elif re.match(r"^\d+$", t):

            nums.append(t)

    if stage is None:

        m = re.search(r"(S\d+)", s, flags=re.IGNORECASE)

        stage = m.group(1).upper() if m else "S?"

    batch = nums[0] if len(nums) >= 1 else "1"

    rep = nums[1] if len(nums) >= 2 else "1"

    sid = f"{cultivar}_{stage}_{batch}_{rep}"

    return {"SampleID": sid, "Cultivar": cultivar, "Stage": stage, "Batch": batch, "Rep": rep}

def clean_raw_csv(in_csv: str, organ: str, out_csv: str, metadata_blacklist: list[str] | None = None) -> pd.DataFrame:

    md = metadata_blacklist or META_BLACKLIST_DEFAULT

    df = pd.read_csv(in_csv)

    sc = _infer_sample_col(df)

    df = df.rename(columns={sc: "SampleID"})

    drop_cols = [c for c in md if c in df.columns]

    if drop_cols:

        df = df.drop(columns=drop_cols)

    meta = df["SampleID"].map(standardize_sampleid).apply(pd.Series)

    df["SampleID"] = meta["SampleID"]

    voc_cols = [c for c in df.columns if c != "SampleID"]

    for c in voc_cols:

        df[c] = pd.to_numeric(df[c], errors="coerce")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)

    return df

def build_clean_ssot(repo: str, leaf_csv: str, peel_csv: str,

                     out_ssot_long: str, out_ssot_cultivar_stage: str,

                     metadata_blacklist: list[str] | None = None):

    repo = str(repo)

    leaf_clean = clean_raw_csv(leaf_csv, "Leaf", str(Path(repo)/"data/raw_clean/GCMS_leaf.clean.csv"), metadata_blacklist)

    peel_clean = clean_raw_csv(peel_csv, "Peel", str(Path(repo)/"data/raw_clean/GCMS_peel.clean.csv"), metadata_blacklist)

    def to_long(df: pd.DataFrame, organ: str) -> pd.DataFrame:

        meta = df["SampleID"].map(standardize_sampleid).apply(pd.Series)

        voc_cols = [c for c in df.columns if c != "SampleID"]

        long = df.melt(id_vars=["SampleID"], value_vars=voc_cols, var_name="VOC", value_name="abundance")

        long = long.merge(meta[["SampleID","Cultivar","Stage","Batch","Rep"]], on="SampleID", how="left")

        long["Organ"] = organ

        return long[["SampleID","Cultivar","Stage","Organ","Batch","Rep","VOC","abundance"]]

    ssot_long = pd.concat([to_long(leaf_clean, "Leaf"), to_long(peel_clean, "Peel")], ignore_index=True)

    Path(out_ssot_long).parent.mkdir(parents=True, exist_ok=True)

    ssot_long.to_parquet(out_ssot_long, index=False)

    vocs = sorted(ssot_long["VOC"].astype(str).unique().tolist())

    grp = ssot_long.groupby(["Cultivar","Stage","Organ","VOC"], dropna=False)["abundance"].mean().reset_index()

    wide = grp.pivot_table(index=["Cultivar","Stage","Organ"], columns="VOC", values="abundance", aggfunc="mean").reset_index()

    wide.columns = [c if isinstance(c,str) else str(c) for c in wide.columns]

    Path(out_ssot_cultivar_stage).parent.mkdir(parents=True, exist_ok=True)

    wide.to_parquet(out_ssot_cultivar_stage, index=False)

def load_ssot_cultivar_stage(path: str) -> pd.DataFrame:

    df = read_table(path)

    for c in ["Cultivar","Stage","Organ"]:

        if c not in df.columns:

            raise ValueError(f"ssot_cultivar_stage missing column {c}: {path}")

    return df

def subset_matrix(ssot_cs: pd.DataFrame, organ: str, stage: str, vocs: list[str] | None = None) -> pd.DataFrame:

    d = ssot_cs[(ssot_cs["Organ"].astype(str)==organ) & (ssot_cs["Stage"].astype(str)==stage)].copy()

    d = d.set_index("Cultivar")

    meta_cols = {"Cultivar","Stage","Organ","Batch","Rep","SampleID"}

    voc_cols = [c for c in d.columns if c not in meta_cols]

    if vocs is not None:

        voc_cols = [c for c in vocs if c in d.columns]

    X = d[voc_cols].apply(pd.to_numeric, errors="coerce")

    return X

def compute_peel_index(vmat: pd.DataFrame, weights: pd.DataFrame, log1p: bool = True) -> pd.Series:

    w = weights.set_index("VOC")["weight"]

    common = [c for c in vmat.columns if c in w.index]

    if not common:

        raise ValueError("No overlapping VOCs between matrix and weights.")

    X = vmat[common].to_numpy(float)

    if log1p:

        X = np.log1p(np.clip(X, a_min=0, a_max=None))

    ww = w.loc[common].to_numpy(float)

    return pd.Series(X @ ww, index=vmat.index, name="peel_index")

def load_pred_vectors_long(path: str) -> pd.DataFrame:

    df = read_table(path)

    ren = {}

    for a,b in [("cultivar","Cultivar"),("stage","Stage"),("organ","Organ"),("voc","VOC"),

                ("y_true","y_true"),("y_pred","y_pred"),("true","y_true"),("pred","y_pred")]:

        if a in df.columns and b not in df.columns:

            ren[a]=b

    df = df.rename(columns=ren)

    need_any = ("VOC" in df.columns)

    if not need_any:

        raise ValueError(f"pred_vectors_long missing VOC column: {path}")

    if "Cultivar" not in df.columns and "SampleID" in df.columns:

        meta = df["SampleID"].map(standardize_sampleid).apply(pd.Series)

        df["Cultivar"] = meta["Cultivar"]

    return df

def predvec_to_matrix(df: pd.DataFrame, value_col: str, organ: str, stage: str) -> pd.DataFrame:

    d = df.copy()

    if "Organ" in d.columns:

        d = d[d["Organ"].astype(str)==organ]

    if "Stage" in d.columns:

        d = d[d["Stage"].astype(str)==stage]

    if value_col not in d.columns:

        cands = [c for c in ["y_true","y_pred"] if c in d.columns]

        if not cands:

            raise ValueError(f"Cannot find prediction value col in pred_vectors_long. Missing {value_col}.")

        value_col = cands[0]

    d["VOC"] = d["VOC"].astype(str)

    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")

    mat = d.pivot_table(index="Cultivar", columns="VOC", values=value_col, aggfunc="mean")

    mat = mat.sort_index()

    mat.columns = mat.columns.astype(str)

    return mat
