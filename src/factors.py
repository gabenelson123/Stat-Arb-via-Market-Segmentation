# src/factors.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class RollingPCAResult:
    date: pd.Timestamp
    U: np.ndarray              # (N_assets, k)
    lam: np.ndarray            # (k,)
    k: int
    explained: float
    valid_cols: list[str]

def standardize_window(W: np.ndarray) -> np.ndarray:
    """
    Standardize columns to mean 0, std 1 (ddof=1).
    W shape: (T, N)
    """
    mu = np.nanmean(W, axis=0, keepdims=True)
    sd = np.nanstd(W, axis=0, ddof=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (W - mu) / sd

def pca_from_standardized(Z: np.ndarray, var_threshold: float = 0.55) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    PCA on correlation matrix of standardized data.
    Returns U (N x k), lam (k,), k, explained.
    """
    T, N = Z.shape
    if T < 2 or N < 2:
        raise ValueError("Need at least T>=2 and N>=2 for PCA.")

    C = (Z.T @ Z) / (T - 1)  # correlation-like
    evals, evecs = np.linalg.eigh(C)      # ascending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    total = float(np.sum(evals))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid eigenvalues in PCA.")

    cum = np.cumsum(evals) / total
    k = int(np.searchsorted(cum, var_threshold) + 1)
    k = max(1, min(k, N))

    U = evecs[:, :k]
    lam = evals[:k]
    explained = float(np.sum(lam) / total)
    return U, lam, k, explained

def run_rolling_pca(
    rets: pd.DataFrame,
    L: int = 252,
    min_coverage: float = 0.80,
    var_threshold: float = 0.55,
) -> list[RollingPCAResult]:
    """
    For each date >= L, run PCA on the last L days using columns with coverage >= min_coverage.
    We then drop any rows with NaN across those columns (same as your notebook).
    """
    if rets is None or rets.empty:
        return []

    cols_order = list(rets.columns)
    out: list[RollingPCAResult] = []
    dates = rets.index

    for i in range(L - 1, len(dates)):
        window = rets.iloc[i - L + 1 : i + 1]

        coverage = window.notna().mean(axis=0)
        valid_cols = [c for c in cols_order if (c in coverage.index and float(coverage[c]) >= min_coverage)]
        if len(valid_cols) < 2:
            continue

        W = window[valid_cols].dropna(axis=0, how="any")
        if W.shape[0] < 2 or W.shape[1] < 2:
            continue

        Z = standardize_window(W.values.astype(float))
        U, lam, k, explained = pca_from_standardized(Z, var_threshold=var_threshold)

        out.append(
            RollingPCAResult(
                date=pd.to_datetime(W.index[-1]),
                U=U,
                lam=lam,
                k=int(k),
                explained=float(explained),
                valid_cols=valid_cols,
            )
        )

    return out

def compute_factor_returns(
    rets: pd.DataFrame,
    rolling: list[RollingPCAResult],
    L: int = 252,
    max_pcs: int = 10,
    min_rows: int = 180,
) -> pd.DataFrame:
    """
    Factor returns f_t = U^T z_t, where z_t is standardized cross-section at time t inside the PCA window.
    Output: DataFrame indexed by date with columns PC1..PCmax.
    """
    if rets is None or rets.empty or not rolling:
        return pd.DataFrame()

    rows: list[dict] = []
    dates = rets.index

    for r in rolling:
        d = pd.to_datetime(r.date)
        if d not in dates:
            continue

        i = dates.get_loc(d)
        win_idx = dates[max(0, i - L + 1) : i + 1]
        W = rets.loc[win_idx, r.valid_cols].dropna(axis=0, how="any")
        if W.shape[0] < min_rows or W.shape[1] < 2:
            continue

        Z = standardize_window(W.values.astype(float))
        z_t = Z[-1, :]  # last standardized cross-section

        k_used = min(r.U.shape[1], max_pcs)
        f_t = (r.U[:, :k_used].T @ z_t)  # shape (k_used,)

        row = {"date": d}
        for j in range(k_used):
            row[f"PC{j+1}"] = float(f_t[j])
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    fac = pd.DataFrame(rows).sort_values("date").set_index("date")
    # Ensure consistent columns PC1..PCmax
    for j in range(1, max_pcs + 1):
        c = f"PC{j}"
        if c not in fac.columns:
            fac[c] = np.nan
    return fac[[f"PC{j}" for j in range(1, max_pcs + 1)]]

