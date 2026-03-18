from __future__ import annotations

import os, time

from pathlib import Path

def find_repo_root(start: str | None = None) -> Path:

    p = Path(start or os.getcwd()).resolve()

    for _ in range(10):

        if (p / "data").exists() and (p / "results").exists():

            return p

        p = p.parent

    return Path(start or os.getcwd()).resolve()

def ts_compact() -> str:

    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def ensure_dir(p: str | Path) -> Path:

    pp = Path(p)

    pp.mkdir(parents=True, exist_ok=True)

    return pp

def default_outdir(repo: Path) -> Path:

    return ensure_dir(repo / "results" / "nc_addons" / ts_compact())
