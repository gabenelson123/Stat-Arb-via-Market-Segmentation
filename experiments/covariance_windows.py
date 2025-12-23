
---

## `experiments/covariance_windows.py`

```python
#!/usr/bin/env python3
"""
Compute covariance matrices on a trailing window:
- sample covariance (pairwise)
- Ledoit–Wolf shrinkage covariance (requires complete rows)

Outputs:
- cov_sample.csv
- cov_ledoitwolf.csv
- variance_hist_sample.png
- variance_hist_ledoitwolf.png
- summary.json
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

from sklearn.covariance import LedoitWolf


# ----------------------------- Bloomberg CSV loader -----------------------------

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
    """
    Loads a Bloomberg wide export like your merged_data.csv:

    row 0: tickers
    row 1: Last Price
    row 2: Dates, PX_LAST, PX_LAST, ...
    row 3+: actual data

    Auto-detects the row where col0 is 'Dates' or 'Date'.
    Returns: prices DataFrame indexed by datetime, columns=tickers, dtype=float
    """
    raw = pandas.read_csv(csv_path, header=None, dtype=str, low_memory=False)

    date_header_row = None
    for i in range(min(30, len(raw))):
        v0 = raw.iat[i, 0]
        v0 = "" if pandas.isna(v0) else str(v0).strip().lower()
        if v0 in {"dates", "date"}:
            date_header_row = i
            break

    if date_header_row is None:
        # Fallback: try normal header CSV
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

    # Align columns safely even if there is an off-by-one mismatch
    n_data_cols = df.shape[1] - 1
    tickers = tickers[:n_data_cols]
    df = df.iloc[:, : 1 + len(tickers)]
    df.columns = ["Date"] + tickers

    df["Date"] = pandas.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    df = df.apply(pandas.to_numeric, errors="coerce")
    df = df.replace([numpy.inf, -numpy.inf], numpy.nan)

    # Merge duplicate names if they exist
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0, sort=False).mean(numeric_only=True).T

    df = df.dropna(how="all").ffill()
    return df


# ----------------------------- Cleaning & returns -----------------------------

def filter_active_columns(
    prices: pandas.DataFrame,
    window: int,
    start_window: int = 5,
    end_window: int = 5,
    drop_name_substr: str | None = "Index",
) -> pandas.DataFrame:
    """
    Keep columns that have at least one observation near the start and end of the window,
    then forward-fill within the window and drop columns still missing.
    """
    px = prices.tail(window).copy()

    has_start = px.head(start_window).notna().any(axis=0)
    has_end = px.tail(end_window).notna().any(axis=0)
    keep = has_start & has_end
    px = px.loc[:, keep.values].copy()

    if drop_name_substr:
        mask = ~pandas.Index(px.columns).str.contains(drop_name_substr, case=False, na=False)
        px = px.loc[:, mask].copy()

    px = px.ffill()

    # After ffill, drop columns still missing anywhere (usually dead names)
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
    rets = rets.dropna(how="all")
    return rets


# ----------------------------- Covariance methods -----------------------------

def sample_covariance(rets: pandas.DataFrame) -> pandas.DataFrame:
    return rets.cov()


def ledoit_wolf_covariance(rets: pandas.DataFrame) -> tuple[pandas.DataFrame | None, float | None]:
    """
    Ledoit–Wolf needs a complete matrix (no NaN). We use complete rows.
    """
    X = rets.dropna(axis=0, how="any")
    if X.shape[0] < 2 or X.shape[1] < 2:
        return None, None
    lw = LedoitWolf().fit(X.values)
    cov = pandas.DataFrame(lw.covariance_, index=X.columns, columns=X.columns)
    return cov, float(lw.shrinkage_)


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


# ----------------------------- CLI -----------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to merged_data.csv (Bloomberg wide export).")
    p.add_argument("--out", default="results/covariance_windows", help="Output directory.")
    p.add_argument("--window", type=int, default=252, help="Trailing window length in trading days.")
    p.add_argument("--universe-n", type=int, default=513, help="Use first N columns after cleaning (0 = all).")
    p.add_argument("--winsor-q", type=float, default=0.005, help="Winsorization tail quantile for returns.")
    p.add_argument("--start-window", type=int, default=5, help="Start-of-window availability check.")
    p.add_argument("--end-window", type=int, default=5, help="End-of-window availability check.")
    p.add_argument("--drop-name-substr", default="Index", help="Drop columns whose name contains this substring (case-insensitive). Use '' to disable.")
    args = p.parse_args()

    out_dir = Path(args.out)
    (out_dir / "matrices").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    drop_sub = args.drop_name_substr if args.drop_name_substr.strip() != "" else None

    print("Loading prices...")
    prices = load_bloomberg_wide_pxlast(args.data)

    print("Filtering active universe...")
    prices = filter_active_columns(
        prices,
        window=args.window,
        start_window=args.start_window,
        end_window=args.end_window,
        drop_name_substr=drop_sub,
    )

    if args.universe_n and args.universe_n > 0:
        prices = prices.iloc[:, : min(args.universe_n, prices.shape[1])].copy()

    print("Computing returns...")
    rets = compute_log_returns(prices, winsor_q=args.winsor_q)

    if rets.shape[0] < 2 or rets.shape[1] < 2:
        raise RuntimeError(f"Not enough data after cleaning: returns shape = {rets.shape}")

    date_range = (str(rets.index.min().date()), str(rets.index.max().date()))
    print(f"Returns: T={rets.shape[0]} rows, N={rets.shape[1]} cols, range={date_range}")

    # Sample covariance
    cov_s = sample_covariance(rets)
    cov_s_path = out_dir / "matrices" / "cov_sample.csv"
    cov_s.to_csv(cov_s_path)
    variance_hist(cov_s, out_dir / "plots" / "variance_hist_sample.png", f"Variance (Sample Cov) — N={rets.shape[1]}, T={rets.shape[0]}")

    # Ledoit–Wolf
    cov_lw, shrink = ledoit_wolf_covariance(rets)
    cov_lw_path = None
    if cov_lw is not None:
        cov_lw_path = out_dir / "matrices" / "cov_ledoitwolf.csv"
        cov_lw.to_csv(cov_lw_path)
        variance_hist(cov_lw, out_dir / "plots" / "variance_hist_ledoitwolf.png", f"Variance (Ledoit–Wolf) — N={cov_lw.shape[0]}, T={rets.dropna().shape[0]}")
    else:
        print("Ledoit–Wolf skipped (not enough complete rows).")

    summary = {
        "data": str(args.data),
        "window": args.window,
        "winsor_q": args.winsor_q,
        "date_range": date_range,
        "prices_shape": [int(prices.shape[0]), int(prices.shape[1])],
        "returns_shape": [int(rets.shape[0]), int(rets.shape[1])],
        "cov_sample_shape": [int(cov_s.shape[0]), int(cov_s.shape[1])],
        "cov_ledoitwolf_shape": None if cov_lw is None else [int(cov_lw.shape[0]), int(cov_lw.shape[1])],
        "ledoitwolf_shrinkage": shrink,
        "paths": {
            "cov_sample": str(cov_s_path),
            "cov_ledoitwolf": None if cov_lw_path is None else str(cov_lw_path),
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done. Wrote:", out_dir)


if __name__ == "__main__":
    main()
