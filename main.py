# main.py
from __future__ import annotations
import os
import pandas as pd

from src.data_loader import load_bloomberg_wide_csv
from src.universe import filter_active_universe
from src.returns import compute_log_returns
from src.factors import run_rolling_pca, compute_factor_returns
from src.residuals import compute_eps_last, build_spread_from_eps
from src.signals import rolling_zscore, alpha_from_z
from src.backtest import run_backtest, BacktestConfig

DATA_PATH = "data/merged_data.csv"
OUT_DIR = "results"

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    prices = load_bloomberg_wide_csv(DATA_PATH)
    prices = filter_active_universe(prices, tail_days=252, max_tail_nan_frac=0.80)

    rets = compute_log_returns(prices, winsor_q=0.005)

    # Rolling PCA + factor returns
    rolling = run_rolling_pca(rets, L=252, min_coverage=0.80, var_threshold=0.55)
    factors = compute_factor_returns(rets, rolling, L=252, max_pcs=8, min_rows=180)

    # Residuals -> spread -> zscore -> alpha
    eps_last = compute_eps_last(rets, factors, lookback=60, min_rows=45)
    X = build_spread_from_eps(eps_last)
    z = rolling_zscore(X, lookback=252, min_periods=126)
    alpha = alpha_from_z(z, clip=3.0)

    # Backtest
    cfg = BacktestConfig(
        roll_win=126,
        cov_recalc_step=5,
        gross=1.0,
        per_name_cap=0.02,
        ridge=1e-6,
        min_assets=50,
    )
    bt = run_backtest(rets, alpha, cfg=cfg)

    bt.to_csv(os.path.join(OUT_DIR, "backtest.csv"))
    print("Saved:", os.path.join(OUT_DIR, "backtest.csv"))
    print(bt.tail())

if __name__ == "__main__":
    main()
