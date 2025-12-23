# src/signals.py
from __future__ import annotations
import numpy as np
import pandas as pd

def rolling_zscore(X: pd.DataFrame, lookback: int = 252, min_periods: int = 126) -> pd.DataFrame:
    mu = X.rolling(window=lookback, min_periods=min_periods).mean()
    sd = X.rolling(window=lookback, min_periods=min_periods).std(ddof=1)
    return (X - mu) / sd

def alpha_from_z(z: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """
    Default mean reversion alpha: alpha = -z (buy negative z, sell positive z).
    """
    a = -z.copy()
    if clip is not None:
        a = a.clip(-clip, clip)
    return a

