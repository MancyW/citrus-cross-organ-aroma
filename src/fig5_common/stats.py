from __future__ import annotations

import numpy as np

import pandas as pd

def rankdata_average_ties(a: np.ndarray) -> np.ndarray:
    """Return average ranks, with ties assigned their average rank."""
    s = pd.Series(np.asarray(a))
    return s.rank(method="average").to_numpy(dtype=float)

def _rankdata(a: np.ndarray) -> np.ndarray:
    return rankdata_average_ties(a)

def spearmanr(x: np.ndarray, y: np.ndarray) -> float:

    x = np.asarray(x, float); y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y)

    if m.sum() < 3:

        return np.nan

    rx = _rankdata(x[m]); ry = _rankdata(y[m])

    if np.std(rx) == 0 or np.std(ry) == 0:

        return np.nan

    return float(np.corrcoef(rx, ry)[0, 1])

def bootstrap_ci_spearman(y_true: np.ndarray, y_pred: np.ndarray, B: int = 2000, seed: int = 0):

    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)

    n = len(y_true)

    rhos = []

    for _ in range(B):

        idx = rng.integers(0, n, size=n)

        rhos.append(spearmanr(y_true[idx], y_pred[idx]))

    rhos = np.asarray(rhos, float)

    return float(np.nanmean(rhos)), float(np.nanquantile(rhos, 0.025)), float(np.nanquantile(rhos, 0.975)), rhos

def loco_align_sign(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:

    y_true = np.asarray(y_true, float)

    y_pred = np.asarray(y_pred, float)

    n = len(y_true)

    out = np.zeros(n, float)

    for i in range(n):

        m = np.ones(n, dtype=bool); m[i] = False

        rho = spearmanr(y_true[m], y_pred[m])

        s = 1.0 if (np.isnan(rho) or rho >= 0) else -1.0

        out[i] = y_pred[i] * s

    return out

def ridge_fit_predict_loco(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):

    X = np.asarray(X, float); y = np.asarray(y, float)

    n, p = X.shape

    yhat = np.zeros(n, float)

    for i in range(n):

        m = np.ones(n, dtype=bool); m[i] = False

        Xtr, ytr = X[m], y[m]

        Xte = X[~m]

        xmu = Xtr.mean(axis=0); ymu = ytr.mean()

        Xc = Xtr - xmu; yc = ytr - ymu

        A = Xc.T @ Xc + alpha * np.eye(p)

        b = Xc.T @ yc

        coef = np.linalg.solve(A, b)

        yhat[i] = float((Xte - xmu) @ coef + ymu)

    return yhat
