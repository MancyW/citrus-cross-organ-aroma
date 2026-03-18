from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

def save_barh_top(df: pd.DataFrame, value_col: str, label_col: str, out_png: str, topk: int = 20, title: str = ""):

    d = df.dropna(subset=[value_col, label_col]).copy()

    d = d.sort_values(value_col, ascending=True).tail(topk)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    plt.figure()

    plt.barh(d[label_col].astype(str), d[value_col].to_numpy())

    plt.title(title)

    plt.tight_layout()

    plt.savefig(out_png, dpi=200)

    plt.close()

def save_hist(x: np.ndarray, out_png: str, title: str = "", xlabel: str = ""):

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    plt.figure()

    plt.hist(x[np.isfinite(x)], bins=30)

    plt.title(title)

    plt.xlabel(xlabel)

    plt.tight_layout()

    plt.savefig(out_png, dpi=200)

    plt.close()
