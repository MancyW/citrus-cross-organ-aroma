from __future__ import annotations

from dataclasses import dataclass

from typing import Optional, List

import numpy as np

from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.preprocessing import StandardScaler

@dataclass

class _OneTargetModel:

    mode: str

    scaler: Optional[StandardScaler]

    clf: Optional[LogisticRegression]

    reg: Optional[Ridge]

    y_mean_log: float

    tau: float

def _safe_log(y: np.ndarray, log1p: bool) -> np.ndarray:

    y = np.asarray(y, dtype=float)

    if log1p:

        return np.log1p(np.clip(y, 0.0, None))

    return np.log(np.clip(y, 1e-12, None))

def _safe_inv_log(z: np.ndarray, log1p: bool) -> np.ndarray:

    z = np.asarray(z, dtype=float)

    if log1p:

        return np.expm1(z)

    return np.exp(z)

def _choose_tau_match_prevalence(p_train: np.ndarray, y_bin: np.ndarray) -> float:

    p_train = np.asarray(p_train, dtype=float)

    y_bin = np.asarray(y_bin, dtype=int)

    pi = float(np.mean(y_bin))

    if pi <= 0.0:

        return 1.0

    if pi >= 1.0:

        return 0.0

    q = 1.0 - pi

    tau = float(np.quantile(p_train, q))

    tau = float(np.clip(tau, 1e-4, 1.0 - 1e-4))

    return tau

def _fit_one_target(

    X: np.ndarray,

    y: np.ndarray,

    *,

    min_pos: int,

    log1p: bool,

    random_state: int,

    clf_C: float,

    clf_max_iter: int,

    ridge_alpha: float,

    p_thresh: float,

    tau_strategy: str,

    class_weight: Optional[str],

) -> _OneTargetModel:

    X = np.asarray(X, dtype=float)

    y = np.asarray(y, dtype=float)

    y_bin = (y > 0).astype(int)

    n = int(y_bin.shape[0])

    n_pos = int(y_bin.sum())

    if n_pos == 0 or n_pos < min_pos:

        return _OneTargetModel(

            mode="all_zero",

            scaler=None,

            clf=None,

            reg=None,

            y_mean_log=0.0,

            tau=1.0,

        )

    if n_pos == n:

        mode = "all_one"

    else:

        mode = "normal"

    scaler = StandardScaler(with_mean=True, with_std=True)

    Xs = scaler.fit_transform(X)

    clf = None

    tau = float(p_thresh)

    if mode == "normal":

        clf = LogisticRegression(

            C=clf_C,

            solver="liblinear",

            max_iter=clf_max_iter,

            random_state=random_state,

            class_weight=class_weight,

        )

        clf.fit(Xs, y_bin)

        if tau_strategy == "match_prevalence":

            p_train = clf.predict_proba(Xs)[:, 1]

            tau = _choose_tau_match_prevalence(p_train, y_bin)

        elif tau_strategy == "fixed":

            tau = float(p_thresh)

        else:

            raise ValueError(f"Unknown tau_strategy: {tau_strategy}")

    elif mode == "all_one":

        tau = 0.0

    idx = (y_bin == 1)

    y_pos = y[idx]

    y_log = _safe_log(y_pos, log1p=log1p)

    y_mean_log = float(np.mean(y_log))

    y_center = y_log - y_mean_log

    reg = Ridge(alpha=ridge_alpha, fit_intercept=False)

    reg.fit(Xs[idx], y_center)

    return _OneTargetModel(

        mode=mode,

        scaler=scaler,

        clf=clf,

        reg=reg,

        y_mean_log=y_mean_log,

        tau=tau,

    )

def _predict_one_target(

    m: _OneTargetModel,

    X: np.ndarray,

    *,

    log1p: bool,

    gate_mode: str,

) -> np.ndarray:

    X = np.asarray(X, dtype=float)

    n = X.shape[0]

    if m.mode == "all_zero":

        return np.zeros(n, dtype=float)

    Xs = m.scaler.transform(X) if m.scaler is not None else X

    if m.mode == "all_one":

        p = np.ones(n, dtype=float)

    else:

        p = m.clf.predict_proba(Xs)[:, 1].astype(float)

    z = m.reg.predict(Xs).astype(float) + m.y_mean_log

    y_pos_hat = _safe_inv_log(z, log1p=log1p)

    y_pos_hat = np.clip(y_pos_hat, 0.0, None)

    if gate_mode == "expected":

        return p * y_pos_hat

    elif gate_mode == "hard":

        return (p >= m.tau).astype(float) * y_pos_hat

    else:

        raise ValueError(f"Unknown gate_mode: {gate_mode}")

class HurdleRegressor:

    def __init__(

        self,

        *,

        min_pos: int = 1,

        log1p: bool = True,

        n_jobs: int = 1,

        random_state: int = 0,

        clf_C: float = 1.0,

        clf_max_iter: int = 300,

        ridge_alpha: float = 1.0,

        p_thresh: float = 0.5,

        class_weight: Optional[str] = "balanced",

        gate_mode: str = "expected",

        tau_strategy: str = "fixed",

    ):

        self.min_pos = int(min_pos)

        self.log1p = bool(log1p)

        self.n_jobs = int(n_jobs)

        self.random_state = int(random_state)

        self.clf_C = float(clf_C)

        self.clf_max_iter = int(clf_max_iter)

        self.ridge_alpha = float(ridge_alpha)

        self.p_thresh = float(p_thresh)

        self.class_weight = class_weight

        self.gate_mode = gate_mode

        self.tau_strategy = tau_strategy

        self.models_: List[_OneTargetModel] = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "HurdleRegressor":

        X = np.asarray(X, dtype=float)

        Y = np.asarray(Y, dtype=float)

        if Y.ndim != 2:

            raise ValueError(f"Y must be 2D (n_samples, n_targets). got {Y.shape}")

        if X.shape[0] != Y.shape[0]:

            raise ValueError(f"X and Y n_samples mismatch: {X.shape[0]} vs {Y.shape[0]}")

        n_targets = Y.shape[1]

        self.models_ = Parallel(n_jobs=self.n_jobs)(

            delayed(_fit_one_target)(

                X,

                Y[:, j],

                min_pos=self.min_pos,

                log1p=self.log1p,

                random_state=self.random_state,

                clf_C=self.clf_C,

                clf_max_iter=self.clf_max_iter,

                ridge_alpha=self.ridge_alpha,

                p_thresh=self.p_thresh,

                tau_strategy=self.tau_strategy,

                class_weight=self.class_weight,

            )

            for j in range(n_targets)

        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        if not self.models_:

            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X, dtype=float)

        preds = Parallel(n_jobs=self.n_jobs)(

            delayed(_predict_one_target)(

                m, X,

                log1p=self.log1p,

                gate_mode=self.gate_mode,

            )

            for m in self.models_

        )

        return np.vstack(preds).T
