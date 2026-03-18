from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

def iter_loco(groups: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:

    groups = np.asarray(groups)

    uniq = np.unique(groups)

    for g in uniq:

        test_idx = np.where(groups == g)[0]

        train_idx = np.where(groups != g)[0]

        yield train_idx, test_idx
