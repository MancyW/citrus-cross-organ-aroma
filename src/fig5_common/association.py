from __future__ import annotations

import numpy as np

import pandas as pd

from .stats import rankdata_average_ties

def _zscore_cols(M: np.ndarray) -> np.ndarray:

    M = M.astype(float)

    mu = np.nanmean(M, axis=0, keepdims=True)

    M0 = M - mu

    sd = np.nanstd(M0, axis=0, ddof=1)

    sd[sd == 0] = np.nan

    return M0 / sd

def pearson_corr_matrix(A: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:

    A0 = _zscore_cols(A.values)

    B0 = _zscore_cols(B.values)

    corr = np.nanmean(A0[:, :, None] * B0[:, None, :], axis=0)

    return pd.DataFrame(corr, index=A.columns, columns=B.columns)

def spearman_corr_matrix(A: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:

    Ar = np.vstack([rankdata_average_ties(A.iloc[:, j].to_numpy(dtype=float)) for j in range(A.shape[1])]).T

    Br = np.vstack([rankdata_average_ties(B.iloc[:, j].to_numpy(dtype=float)) for j in range(B.shape[1])]).T

    A0 = _zscore_cols(Ar)

    B0 = _zscore_cols(Br)

    corr = np.nanmean(A0[:, :, None] * B0[:, None, :], axis=0)

    return pd.DataFrame(corr, index=A.columns, columns=B.columns)

def build_leaf_to_peel_links(

    leaf_mat: pd.DataFrame,

    peel_mat: pd.DataFrame,

    leaf_features: list[str],

    peel_features: list[str],

    min_abs_corr: float = 0.25,

    max_links_per_leaf: int = 10,

    method: str = "spearman",

) -> pd.DataFrame:

    L = leaf_mat[leaf_features].copy()

    P = peel_mat[peel_features].copy()

    if method == "pearson":

        corr = pearson_corr_matrix(L, P)

    elif method == "spearman":

        corr = spearman_corr_matrix(L, P)

    else:

        raise ValueError("method must be 'pearson' or 'spearman'")

    links = corr.stack(dropna=False).reset_index()

    links.columns = ["leaf_feature", "peel_feature", "corr"]

    links["abs_corr"] = links["corr"].abs()

    links = links[np.isfinite(links["abs_corr"]) & (links["abs_corr"] >= float(min_abs_corr))].copy()

    if links.empty:

        return pd.DataFrame(columns=["leaf_feature", "peel_feature", "corr", "abs_corr"])

    links = links.sort_values(["leaf_feature", "abs_corr"], ascending=[True, False])

    links = links.groupby("leaf_feature", as_index=False).head(int(max_links_per_leaf))

    links = links.sort_values("abs_corr", ascending=False).reset_index(drop=True)

    return links
