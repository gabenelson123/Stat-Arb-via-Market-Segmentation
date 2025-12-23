# src/universe.py
from __future__ import annotations
import pandas as pd

def filter_active_universe(
    prices: pd.DataFrame,
    tail_days: int = 252,
    max_tail_nan_frac: float = 0.80,
) -> pd.DataFrame:
    """
    Keep tickers that have "enough" data in the last tail_days.
    This matches your notebook logic: drop "dead" names (delisted, acquired, etc.).

    max_tail_nan_frac = 0.80 means "drop if >80% missing in last year".
    """
    if prices.empty:
        return prices

    tail = prices.tail(tail_days) if len(prices) >= tail_days else prices
    nan_tail = tail.isna().mean(axis=0)  # fraction NaN in tail window
    keep = nan_tail[nan_tail < max_tail_nan_frac].index
    return prices.loc[:, keep]

