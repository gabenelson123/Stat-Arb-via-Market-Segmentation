#!/usr/bin/env python3
"""
Random Matrix Theory (RMT) eigenvalue cleaning of correlation/covariance.

Pipeline:
- load Bloomberg-wide prices
- pick active universe on last window
- compute log returns (winsorized)
- (optional) convert to AR(1) residuals per name
- build correlation, eigen-decompose
- MP bulk cutoff: lambda_plus = (1 + sqrt(q))^2 where q=N/T
- replace bulk eigenvalues with their mean
- rebuild cleaned correlation and map to covariance

Outputs:
- cov_rmt.csv
- variance_hist_rmt.png
- info.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as numpy
import pandas as pandas

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- copy of loader + cleaning (kept local so experiments are standalone) ----

def _make_unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        n = (n or "").strip()
        if n == "":
            n = "col"
        if n in seen:
            seen[n] += 1
            out.append(f"{n}__{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def load_bloomberg_wide_pxlast(csv_path: str | Path) -> pandas.DataFrame:
    raw = pandas.read_csv(csv_path, header=None, dtype=str, low_memory=False)

    date_header_row = None
    for i in range(min(30, len(raw))):
        v0 = raw.iat[i, 0]
        v0 = "" if pandas.isna(v0) else str(v0).strip().lower()
        if v0 in {"dates", "date"}:
            date_header_row = i
            break

    if date_header_row is None:
        df = pandas.read_csv(csv_path, low_memory=False)
        date_col = df.columns[0]
        df[date_col] = pandas.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        df = df.apply(pandas.to_numeric, errors="coerce")
        return df.dropna(how="all").ffill()

    tickers_row = max(0, date_header_row - 2)
    tickers = raw.iloc[tickers_row, 1:].tolist()
    tickers = _make_unique([(t if pandas.notna(t) else "") for t in tickers])

    df = raw.iloc[date_header_row + 1 :].copy()
    df = df.rename(columns={0: "Date"})
    n_data_cols = df.shape[1] - 1
    tickers = tickers[:n_data_cols]
    df = df.iloc[:, : 1 + len(tickers)]
    df.columns = ["Date"] + tickers

    df["Date"] = pandas.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = df.apply(pandas.to_numeric, errors="coerce")
    df = df.replace([numpy.inf, -numpy.inf], numpy.nan)

    if df.columns.duplicated().any():
        df = df.T.groupby(level=0, sort=False).mean(numeric_only=True).T

    return df.dropna(how="all").ffill()


def filter_active_columns(
    prices: pandas.DataFrame,
    window: int,
    start_window: int = 5,
    end_window: int = 5,
    drop_name_substr: str | None = "Index",
) -> pandas.DataFrame:
    px = prices.tail(window).copy()

    has_start = px.head(start_window).notna().any(axis=0)
    has_end = px.tail(end_window).notna().any(axis=0)
    px = px.loc[:, (has_start & has_end).values].copy()

    if drop_name_substr:
        mask = ~pandas.Index(px.columns).str.contains(drop_name_substr, case=False, na=False)
        px = px.loc[:, mask].copy()

    px = px.ffill()
    px = px.dropna(axis=1, how="any")
    return px


def winsorize_returns(rets: pandas.DataFrame, q: float) -> pandas.DataFrame:
    if q is None or q <= 0:
        return rets
    ql = rets.quantile(q)
    qh = rets.quantile(1 - q)
    return rets.clip(lower=ql, upper=qh, axis=1)


def compute_log_returns(prices: pandas.DataFrame, winsor_q: float) -> pandas.DataFrame:
    rets = numpy.log(prices).diff()
    rets = winsorize_returns(rets, winsor_q)
    return rets.dropna(how="all")


def ar1_residuals(rets: pandas.DataFrame) -> pandas.DataFrame:
    """
    Fit per-column AR(1): r_t = phi r_{t-1} + eps_t (no intercept)
    phi = cov(r_t, r_{t-1}) / var(r_{t-1}) computed on available pairs
    """
    r = rets.copy()
    r_lag = r.shift(1)

    # require pairwise aligned rows for each column; we do it columnwise
    out = pandas.DataFrame(index=r.index, columns=r.columns, dtype=float)
    for c in r.columns:
        y = r[c]
        x = r_lag[c]
        df = pandas.concat([y, x], axis=1).dropna()
        if len(df) < 10:
            continue
        yv = df.iloc[:, 0].values
        xv = df.iloc[:, 1].values
        den = numpy.var(xv, ddof=1)
        if not numpy.isfinite(den) or den <= 0:
            continue
        phi = numpy.cov(yv, xv, ddof=1)[0, 1] / den
        # residuals where both exist
        out.loc[df.index, c] = yv - phi * xv
    return out.dropna(how="all")


def rmt_clean_cov(rets: pandas.DataFrame) -> tuple[pandas.DataFrame, dict]:
    """
    RMT eigenvalue cleaning on correlation, then map back to covariance.
    """
    std = rets.std(ddof=1)
    valid_cols = std[std > 0].index
    R = rets[valid_cols].copy()

    X = (R - R.mean()) / R.std(ddof=1)
    X = X.dropna(axis=0, how="any")
    T, N = X.shape
    if T < 2 or N < 2:
        raise RuntimeError(f"Insufficient data: T={T}, N={N}")

    C = numpy.corrcoef(X.values, rowvar=False)
    eigvals, eigvecs = numpy.linalg.eigh(C)  # ascending
    q = N / T
    lambda_plus = (1.0 + numpy.sqrt(q)) ** 2

    bulk_mask = eigvals <= lambda_plus
    bulk_mean = float(eigvals[bulk_mask].mean()) if bulk_mask.any() else float("nan")

    eigvals_clean = eigvals.copy()
    if bulk_mask.any():
        eigvals_clean[bulk_mask] = bulk_mean

    C_clean = eigvecs @ numpy.diag(eigvals_clean) @ eigvecs.T

    # Map back to covariance
    s = R.std(ddof=1).values
    D = numpy.diag(s)
    Sigma_clean = D @ C_clean @ D

    info = {
        "T": int(T),
        "N": int(N),
        "q": float(q),
        "lambda_plus": float(lambda_plus),
        "num_bulk": int(bulk_mask.sum()),
        "bulk_mean": bulk_mean,
    }
    return pandas.DataFrame(Sigma_clean, index=valid_cols, columns=valid_cols), info


def variance_hist(cov: pandas.DataFrame, out_png: Path, title: str) -> None:
    d = numpy.diag(cov.values)
    d = d[numpy.isfinite(d)]
    plt.figure(figsize=(9, 5.5))
    plt.hist(d, bins=40)
    plt.title(title)
    plt.xlabel("Variance")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="results/rmt_cleaning")
    p.add_argument("--window", type=int, default=252)
    p.add_argument("--universe-n", type=int, default=513, help="Use first N columns after cleaning (0 = all).")
    p.add_argument("--winsor-q", type=float, default=0.005)
    p.add_argument("--use-ar1-residuals", action="store_true")
    p.add_argument("--start-window", type=int, default=5)
    p.add_argument("--end-window", type=int, default=5)
    p.add_argument("--drop-name-substr", default="Index")
    args = p.parse_args()

    out_dir = Path(args.out)
    (out_dir / "matrices").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    drop_sub = args.drop_name_substr.strip()
    drop_sub = drop_sub if drop_sub != "" else None

    print("Loading prices...")
    prices = load_bloomberg_wide_pxlast(args.data)
    prices = filter_active_columns(
        prices,
        window=args.window,
        start_window=args.start_window,
        end_window=args.end_window,
        drop_name_substr=drop_sub,
    )

    if args.universe_n and args.universe_n > 0:
        prices = prices.iloc[:, : min(args.universe_n, prices.shape[1])].copy()

    rets = compute_log_returns(prices, winsor_q=args.winsor_q)

    if args.use_ar1_residuals:
        print("Using AR(1) residuals...")
        rets_for_cov = ar1_residuals(rets)
    else:
        rets_for_cov = rets

    print("RMT cleaning...")
    cov_rmt, info = rmt_clean_cov(rets_for_cov)

    cov_path = out_dir / "matrices" / "cov_rmt.csv"
    cov_rmt.to_csv(cov_path)

    plot_path = out_dir / "plots" / "variance_hist_rmt.png"
    variance_hist(cov_rmt, plot_path, f"Variance (RMT Cleaned) — N={cov_rmt.shape[0]}, T≈{info['T']}")

    with open(out_dir / "info.json", "w") as f:
        json.dump(
            {
                "data": args.data,
                "window": args.window,
                "winsor_q": args.winsor_q,
                "use_ar1_residuals": bool(args.use_ar1_residuals),
                "rmt": info,
                "paths": {"cov_rmt": str(cov_path), "variance_hist": str(plot_path)},
            },
            f,
            indent=2,
        )

    print("Done. Wrote:", out_dir)


if __name__ == "__main__":
    main()

