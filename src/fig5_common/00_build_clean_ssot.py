from __future__ import annotations

import argparse

from pathlib import Path

from nc_scripts.common.paths import find_repo_root

from nc_scripts.common import dataio

def main():

    ap = argparse.ArgumentParser(description="Build CLEAN SSOT from raw CSVs (drops metadata cols, standardizes SampleID).")

    ap.add_argument("--repo", default=None)

    ap.add_argument("--leaf_csv", default=None)

    ap.add_argument("--peel_csv", default=None)

    ap.add_argument("--out_ssot_long", default=None)

    ap.add_argument("--out_ssot_cultivar_stage", default=None)

    ap.add_argument("--metadata_blacklist", nargs="*", default=dataio.META_BLACKLIST_DEFAULT)

    args = ap.parse_args()

    repo = find_repo_root(args.repo)

    leaf = args.leaf_csv or str(repo/"data/raw/GCMS_leaf.csv")

    peel = args.peel_csv or str(repo/"data/raw/GCMS_peel.csv")

    out_long = args.out_ssot_long or str(repo/"data/ssot/ssot_long.clean.parquet")

    out_cs = args.out_ssot_cultivar_stage or str(repo/"data/ssot/ssot_cultivar_stage.clean.parquet")

    dataio.build_clean_ssot(str(repo), leaf, peel, out_long, out_cs, args.metadata_blacklist)

    print(f"[OK] wrote: {out_long}")

    print(f"[OK] wrote: {out_cs}")

if __name__ == "__main__":

    main()
