# src/data_loader.py
from __future__ import annotations
import pandas as pd
import numpy as np

def _make_unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        n = (n or "").strip()
        if n == "":
            n = "col"
        k = seen.get(n, 0)
        out.append(n if k == 0 else f"{n}__{k}")
        seen[n] = k + 1
    return out

def load_bloomberg_wide_csv(path: str) -> pd.DataFrame:
    """
    Load a Bloomberg wide export with meta rows (tickers + 'Last Price' + 'PX_LAST')
    and return prices indexed by date with float columns.
    """
    raw = pd.read_csv(path, header=None, dtype=str)

    # Heuristic: tickers are usually row 0 or row 1; choose the row with most non-empty cells after col0
    cand_rows = [0, 1, 2]
    best_row = max(cand_rows, key=lambda r: raw.iloc[r, 1:].notna().sum())
    tickers = raw.iloc[best_row, 1:].tolist()
    tickers = _make_unique([(t or "").strip() for t in tickers])

    # Find first row where first column parses as date for most rows
    col0 = raw.iloc[:, 0].astype(str)
    parsed = pd.to_datetime(col0, errors="coerce")
    date_start = int(np.argmax(parsed.notna().values))  # first parseable date row

    df = raw.iloc[date_start:, :].copy()
    df = df.rename(columns={0: "Date"})
    df.columns = ["Date"] + tickers[: df.shape[1] - 1]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c].str.replace(",", "").str.strip(), errors="coerce")

    # Forward fill only (avoid look-ahead)
    df = df.dropna(how="all").ffill()

    # Drop constant columns
    nunq = df.nunique(dropna=True)
    df = df.loc[:, nunq > 1]

    return df

