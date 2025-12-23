# src/returns.py
from __future__ import annotations
import numpy as np
import pandas as pd

def compute_log_returns(prices: pd.DataFrame, winsor_q: float = 0.005) -> pd.DataFrame:
    r = np.log(prices).diff()

    # Winsorize per column (damps spikes)
    ql = r.quantile(winsor_q)
    qh = r.quantile(1.0 - winsor_q)
    r = r.clip(lower=ql, upper=qh, axis=1)

    return r.dropna(how="all")

