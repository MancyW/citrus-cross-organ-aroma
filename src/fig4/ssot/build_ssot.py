from __future__ import annotations

import argparse

import json

from pathlib import Path

from typing import List, Dict

import pandas as pd

import yaml

def read_yaml(path: str | Path) -> dict:

    path = Path(path)

    with path.open("r", encoding="utf-8") as f:

        return yaml.safe_load(f)

def ensure_parent(p: str | Path):

    Path(p).parent.mkdir(parents=True, exist_ok=True)

def infer_voc_cols(df: pd.DataFrame, meta_cols: List[str]) -> List[str]:

    return [c for c in df.columns if c not in meta_cols]

def coerce_numeric(df: pd.DataFrame, voc_cols: List[str]) -> pd.DataFrame:

    df[voc_cols] = df[voc_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return df

def align_to_union(df: pd.DataFrame, union_vocs: List[str]) -> pd.DataFrame:

    for c in union_vocs:

        if c not in df.columns:

            df[c] = 0.0

    return df[union_vocs]

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--config", default="configs/base.yaml")

    args = ap.parse_args()

    cfg = read_yaml(args.config)

    paths = cfg["paths"]

    cols = cfg["cols"]

    leaf_path = paths["leaf_table"]

    peel_path = paths["peel_table"]

    out_long = paths["ssot_long"]

    out_cs = paths["ssot_cultivar_stage"]

    sample_id = cols.get("sample_id", "SampleID")

    cultivar = cols.get("cultivar", "Cultivar")

    organ = cols.get("organ", "Organ")

    stage = cols.get("stage", "Stage")

    batch = cols.get("batch", "Batch")

    rep = cols.get("rep", "Rep")

    leaf = pd.read_csv(leaf_path)

    peel = pd.read_csv(peel_path)

    required = [cultivar, organ, stage, batch, rep]

    for name, df in [("leaf", leaf), ("peel", peel)]:

        miss = [c for c in required if c not in df.columns]

        if miss:

            raise KeyError(f"[{name}] missing required columns: {miss}. Found: {df.columns.tolist()[:12]} ...")

    has_sample = (sample_id in leaf.columns) and (sample_id in peel.columns)

    leaf["PairID"] = (

        leaf[cultivar].astype(str) + "_" +

        leaf[stage].astype(str) + "_" +

        leaf[batch].astype(int).astype(str) + "_" +

        leaf[rep].astype(int).astype(str)

    )

    peel["PairID"] = (

        peel[cultivar].astype(str) + "_" +

        peel[stage].astype(str) + "_" +

        peel[batch].astype(int).astype(str) + "_" +

        peel[rep].astype(int).astype(str)

    )

    leaf[organ] = "Leaf"

    peel[organ] = "Peel"

    meta_cols = (["PairID"] + required + ([sample_id] if has_sample else []))

    leaf_vocs = infer_voc_cols(leaf, meta_cols)

    peel_vocs = infer_voc_cols(peel, meta_cols)

    union_vocs = sorted(list(set(leaf_vocs) | set(peel_vocs)))

    if len(union_vocs) == 0:

        raise ValueError("No VOC columns detected. Check your input tables.")

    leaf = coerce_numeric(leaf, leaf_vocs)

    peel = coerce_numeric(peel, peel_vocs)

    leaf_union = align_to_union(leaf[leaf_vocs], union_vocs)

    peel_union = align_to_union(peel[peel_vocs], union_vocs)

    base_cols = ["PairID", cultivar, organ, stage, batch, rep]

    if has_sample:

        base_cols = ["PairID", sample_id, cultivar, organ, stage, batch, rep]

    ssot_long = pd.concat(

        [

            pd.concat([leaf[base_cols].reset_index(drop=True), leaf_union.reset_index(drop=True)], axis=1),

            pd.concat([peel[base_cols].reset_index(drop=True), peel_union.reset_index(drop=True)], axis=1),

        ],

        ignore_index=True

    )

    gcols = [cultivar, stage, organ]

    ssot_cs = ssot_long.groupby(gcols, as_index=False)[union_vocs].mean()

    n_df = ssot_long.groupby(gcols, as_index=False).size().rename(columns={"size": "n_samples"})

    ssot_cs = ssot_cs.merge(n_df, on=gcols, how="left")

    ensure_parent(out_long)

    ensure_parent(out_cs)

    ssot_long.to_parquet(out_long, index=False)

    ssot_cs.to_parquet(out_cs, index=False)

    manifest = {

        "inputs": {"leaf_table": str(leaf_path), "peel_table": str(peel_path)},

        "outputs": {"ssot_long": str(out_long), "ssot_cultivar_stage": str(out_cs)},

        "n_rows_leaf": int(leaf.shape[0]),

        "n_rows_peel": int(peel.shape[0]),

        "n_rows_ssot_long": int(ssot_long.shape[0]),

        "n_rows_ssot_cultivar_stage": int(ssot_cs.shape[0]),

        "n_vocs_union": int(len(union_vocs)),

        "cultivar_n": int(ssot_long[cultivar].nunique()),

        "stages": sorted(ssot_long[stage].unique().tolist()),

        "organs": sorted(ssot_long[organ].unique().tolist()),

        "has_sample_id": bool(has_sample),

    }

    manifest_path = Path(out_long).with_suffix(".manifest.json")

    with manifest_path.open("w", encoding="utf-8") as f:

        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("[OK] SSOT built:")

    print("  -", out_long)

    print("  -", out_cs)

    print("  -", str(manifest_path))

    print("[INFO] union VOCs:", len(union_vocs), "| cultivars:", manifest["cultivar_n"], "| stages:", manifest["stages"])

if __name__ == "__main__":

    main()
