from __future__ import annotations

import os

import random

from typing import Optional

import numpy as np

def set_global_seed(seed: int = 42, deterministic: bool = False) -> None:

    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    np.random.seed(seed)

    if deterministic:

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
