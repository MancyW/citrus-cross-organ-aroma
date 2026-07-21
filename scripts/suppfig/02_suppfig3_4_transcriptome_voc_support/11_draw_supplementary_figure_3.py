#!/usr/bin/env python3
"""Draw Supplementary Fig. 3 from the processed transcriptome/VOC support tables."""
from pathlib import Path
import argparse
import importlib.util

HERE = Path(__file__).resolve().parent
SHARED = HERE / "10_draw_supplementary_transcriptome_figures.py"
_spec = importlib.util.spec_from_file_location("supplementary_transcriptome_draw", SHARED)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="Directory containing processed transcriptome/VOC support tables.")
    parser.add_argument("--out", type=str, default="figures/supplementaryfig", help="Output directory.")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    _mod.draw_supplementary_fig3(Path(args.base), Path(args.out))

if __name__ == "__main__":
    main()
