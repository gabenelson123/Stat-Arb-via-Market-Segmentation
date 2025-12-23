# src/backtest.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .covariance import ledoit_wolf_cov
from .portfolio import solve_mean_variance_weights

@dataclass(frozen=True)
class BacktestConfig:
    roll_win: int = 126            # risk window length
    cov_recalc_step: int = 5       # recompute covariance every N days
    gross: float = 1.0
    per_name_cap: float = 0.02
    ridge: float = 0.0
    min_assets: int = 50

def run_backtest(
    rets: pd.DataFrame,      # log returns (dates x tickers)
    alpha: pd.DataFrame,     # same index/cols; alpha_t used to form w_t
    cfg: BacktestConfig = BacktestConfig(),
) -> pd.DataFrame:
    """
    For each date t, form weights from alpha[t] and LW covariance using last cfg.roll_win returns,
    then compute next-day PnL using returns at t+1.

    Output: DataFrame with pnl, gross_exposure, n_names.
    """
    if rets.empty or alpha.empty:
        return pd.DataFrame()

    idx = rets.index.intersection(alpha.index).sort_values()
    rets = rets.reindex(idx)
    alpha = alpha.reindex(idx)

    # We need t and t+1 for PnL
    start = cfg.roll_win
    end = len(idx) - 1
    if end <= start:
        return pd.DataFrame()

    cov_cache = None
    cov_cache_until = -1

    rows = []
    for t_i in range(start, end):
        t = idx[t_i]
        t_next = idx[t_i + 1]

        # alpha at t (trade at close t, realize on t+1)
        a_t = alpha.loc[t].dropna()
        if a_t.shape[0] < cfg.min_assets:
            rows.append({"date": t, "pnl": np.nan, "gross": np.nan, "n": int(a_t.shape[0])})
            continue

        # covariance window ending at t
        win = rets.iloc[t_i - cfg.roll_win + 1 : t_i + 1]
        # intersect columns early for speed
        win = win.loc[:, win.columns.intersection(a_t.index)]

        # cache LW covariance
        if (cov_cache is None) or (t_i > cov_cache_until):
            cov_cache = ledoit_wolf_cov(win)
            cov_cache_until = t_i + cfg.cov_recalc_step

        cov = cov_cache
        if cov is None or cov.shape[0] < cfg.min_assets:
            rows.append({"date": t, "pnl": np.nan, "gross": np.nan, "n": int(a_t.shape[0])})
            continue

        # weights
        w_t = solve_mean_variance_weights(
            cov=cov,
            alpha=a_t,
            gross=cfg.gross,
            per_name_cap=cfg.per_name_cap,
            ridge=cfg.ridge,
        )
        if w_t.empty or w_t.shape[0] < cfg.min_assets:
            rows.append({"date": t, "pnl": np.nan, "gross": np.nan, "n": int(w_t.shape[0])})
            continue

        r_next = rets.loc[t_next].reindex(w_t.index).dropna()
        w_eff = w_t.reindex(r_next.index).fillna(0.0)

        pnl = float((w_eff * r_next).sum())
        gross = float(w_eff.abs().sum())

        rows.append({"date": t, "pnl": pnl, "gross": gross, "n": int(w_eff.shape[0])})

    out = pd.DataFrame(rows).set_index("date").sort_index()
    out["cum_pnl"] = out["pnl"].cumsum()
    return out

