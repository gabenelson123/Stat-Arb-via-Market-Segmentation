# src/covariance.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def ledoit_wolf_cov(returns_window: pd.DataFrame) -> pd.DataFrame | None:
    """
    Ledoit–Wolf covariance on complete rows only.
    Returns None if insufficient data.
    """
    X = returns_window.dropna(axis=0, how="any")
    if X.shape[0] < 2 or X.shape[1] < 2:
        return None
    lw = LedoitWolf().fit(X.values)
    return pd.DataFrame(lw.covariance_, index=X.columns, columns=X.columns)

def sample_cov(returns_window: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise sample covariance (pandas handles pairwise non-missing).
    """
    return returns_window.cov()

