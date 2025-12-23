# src/residuals.py
from __future__ import annotations
import numpy as np
import pandas as pd

def compute_eps_last(
    rets: pd.DataFrame,
    factor_returns: pd.DataFrame,
    lookback: int = 60,
    min_rows: int = 45,
) -> pd.DataFrame:
    """
    For each date t, regress each stock's last `lookback` returns on factor returns (with intercept)
    and store residual epsilon_{i,t} at t.

    Returns: DataFrame (dates x tickers) of eps_last.
    """
    if rets.empty or factor_returns.empty:
        return pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)

    dates = rets.index.intersection(factor_returns.index).sort_values()
    tickers = list(rets.columns)

    pc_cols = [c for c in factor_returns.columns if c.upper().startswith("PC")]
    pc_cols = sorted(pc_cols, key=lambda x: int("".join([ch for ch in x if ch.isdigit()]) or 0))
    if not pc_cols:
        raise ValueError("factor_returns must have PC* columns.")

    eps_rows: list[dict] = []

    for t_idx in range(lookback - 1, len(dates)):
        t = dates[t_idx]
        win = dates[t_idx - lookback + 1 : t_idx + 1]

        Fw = factor_returns.reindex(win)[pc_cols]
        row: dict = {"date": t}

        # If factors too thin, write NaNs for all names
        if Fw.dropna().shape[0] < min_rows:
            for name in tickers:
                row[name] = np.nan
            eps_rows.append(row)
            continue

        # Each stock regression (simple and clear; can be optimized later)
        for name in tickers:
            y = rets.reindex(win)[name]
            df_join = pd.concat([y, Fw], axis=1).dropna()
            if len(df_join) < min_rows or t not in df_join.index:
                row[name] = np.nan
                continue

            yv = df_join[name].values.astype(float)
            X = df_join[pc_cols].values.astype(float)
            Xc = np.column_stack([np.ones(len(X)), X])  # intercept

            beta = np.linalg.pinv(Xc) @ yv
            ft = df_join.loc[t, pc_cols].values.astype(float)
            yhat_t = float(beta[0] + ft @ beta[1:])
            row[name] = float(y.loc[t] - yhat_t)

        eps_rows.append(row)

    eps = pd.DataFrame(eps_rows).set_index("date")
    # reindex to full date index (optional)
    eps = eps.reindex(dates)
    return eps[tickers]

def build_spread_from_eps(eps_last: pd.DataFrame) -> pd.DataFrame:
    """
    X_i(t) = cumulative sum of eps_last[i]. NaNs treated as 0 to keep continuity.
    """
    if eps_last.empty:
        return eps_last
    return eps_last.fillna(0.0).cumsum()

