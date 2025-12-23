# src/portfolio.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _dollar_neutral(w: pd.Series) -> pd.Series:
    if w.empty:
        return w
    return w - w.mean()

def _scale_to_gross(w: pd.Series, gross: float) -> pd.Series:
    s = float(w.abs().sum())
    if s <= 0 or not np.isfinite(s):
        return w * 0.0
    return (gross / s) * w

def solve_mean_variance_weights(
    cov: pd.DataFrame,
    alpha: pd.Series,
    gross: float = 1.0,
    per_name_cap: float | None = None,
    ridge: float = 0.0,
) -> pd.Series:
    """
    Compute w* = (Sigma + ridge I)^{-1} alpha, then cap, dollar-neutral, scale to gross.
    """
    common = cov.columns.intersection(alpha.index)
    if len(common) < 2:
        return pd.Series(dtype=float)

    S = cov.loc[common, common].values.astype(float)
    a = alpha.loc[common].values.astype(float)

    if ridge and ridge > 0:
        S = S + ridge * np.eye(S.shape[0])

    try:
        x = np.linalg.solve(S, a)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(S, a, rcond=None)[0]

    w = pd.Series(x, index=common)

    if per_name_cap is not None:
        w = w.clip(-per_name_cap, per_name_cap)

    w = _dollar_neutral(w)
    w = _scale_to_gross(w, gross=gross)
    return w

