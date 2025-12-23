#!/usr/bin/env python3
"""
Graphical Lasso on a correlation subset, then visualize the precision sparsity pattern.

Inputs:
- Bloomberg-wide prices -> returns -> correlation
- subset by:
  - firstN
  - topN (highest average absolute correlation)
  - range (column slice)

Outputs:
- edges.csv (edge list from precision nonzeros)
- graph.html (plotly interactive)
- info.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as numpy
import pandas as pandas

from sklearn.covariance import graphical_lasso

# Optional visualization deps
try:
    import networkx as nx
    import plotly.graph_objects as go
except Exception:
    nx = None
    go = None


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


def compute_log_returns(prices: pandas.DataFrame, winsor_q: float) -> pandas.DataFrame:
    rets = numpy.log(prices).diff()
    if winsor_q and winsor_q > 0:
        ql = rets.quantile(winsor_q)
        qh = rets.quantile(1 - winsor_q)
        rets = rets.clip(lower=ql, upper=qh, axis=1)
    return rets.dropna(how="all")


def pick_subset(corr: pandas.DataFrame, mode: str, N: int, r0: int, r1: int) -> list[str]:
    if mode == "firstN":
        return list(corr.columns[:N])
    if mode == "topN":
        avg_abs = corr.abs().copy()
        numpy.fill_diagonal(avg_abs.values, numpy.nan)
        return avg_abs.mean(axis=1).sort_values(ascending=False).head(N).index.tolist()
    if mode == "range":
        return list(corr.columns[r0 : r1 + 1])
    raise ValueError("subset-mode must be one of: firstN, topN, range")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="results/graphical_lasso")
    p.add_argument("--window", type=int, default=252)
    p.add_argument("--universe-n", type=int, default=800)
    p.add_argument("--winsor-q", type=float, default=0.005)
    p.add_argument("--subset-mode", default="topN", choices=["firstN", "topN", "range"])
    p.add_argument("--N", type=int, default=150)
    p.add_argument("--range-start", type=int, default=0)
    p.add_argument("--range-end", type=int, default=149)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--tol", type=float, default=2e-4)
    p.add_argument("--enet-tol", type=float, default=1e-4)
    p.add_argument("--edge-abs-tol", type=float, default=1e-12)
    p.add_argument("--drop-isolates", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop-name-substr", default="Index")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading prices...")
    prices = load_bloomberg_wide_pxlast(args.data)
    prices = filter_active_columns(prices, window=args.window, drop_name_substr=args.drop_name_substr or None)
    if args.universe_n and args.universe_n > 0:
        prices = prices.iloc[:, : min(args.universe_n, prices.shape[1])].copy()

    rets = compute_log_returns(prices, winsor_q=args.winsor_q)
    rets = rets.dropna(axis=0, how="any")  # correlation needs complete rows for stability
    if rets.shape[0] < 10 or rets.shape[1] < 2:
        raise RuntimeError(f"Not enough complete data: returns shape={rets.shape}")

    corr = rets.corr()
    corr = 0.5 * (corr + corr.T)

    subset_cols = pick_subset(corr, args.subset_mode, args.N, args.range_start, args.range_end)
    subset_cols = [c for c in subset_cols if c in corr.columns]
    C = corr.loc[subset_cols, subset_cols].to_numpy(float)
    C = 0.5 * (C + C.T)

    # Graphical lasso with jitter retries (your pattern)
    C_hat = None
    Theta = None
    costs = None
    last_err = None
    for jit in (0.0, 1e-10, 1e-9, 1e-8):
        try:
            Cj = C + jit * numpy.eye(C.shape[0])
            C_hat, Theta, costs = graphical_lasso(
                emp_cov=Cj,
                alpha=args.alpha,
                max_iter=args.max_iter,
                tol=args.tol,
                enet_tol=args.enet_tol,
                return_costs=True,
                verbose=True,
            )
            last_err = None
            break
        except Exception as e:
            last_err = str(e)

    if Theta is None:
        raise RuntimeError(f"Graphical Lasso failed. Last error: {last_err}")

    A = Theta.copy()
    numpy.fill_diagonal(A, 0.0)
    mask = numpy.abs(A) > args.edge_abs_tol

    # Edge list
    edges = []
    cols = subset_cols
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if mask[i, j]:
                edges.append((cols[i], cols[j], float(Theta[i, j])))

    edges_df = pandas.DataFrame(edges, columns=["u", "v", "theta_ij"])
    edges_path = out_dir / "edges.csv"
    edges_df.to_csv(edges_path, index=False)

    info = {
        "data": args.data,
        "window": args.window,
        "winsor_q": args.winsor_q,
        "subset_mode": args.subset_mode,
        "subset_size": int(len(subset_cols)),
        "alpha": float(args.alpha),
        "num_edges": int(len(edges)),
        "complete_rows_T": int(rets.shape[0]),
    }

    # Optional interactive graph
    html_path = None
    if nx is not None and go is not None:
        G = nx.Graph()
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        if args.drop_isolates:
            G = G.subgraph([n for n, d in G.degree() if d > 0]).copy()

        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, seed=args.seed)
            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="skip", opacity=0.35)

            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            deg = dict(G.degree())
            node_text = [f"{n} | degree={deg[n]}" for n in G.nodes()]

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hovertext=node_text,
                hoverinfo="text",
                marker=dict(size=10),
            )

            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                title=f"Graphical Lasso (alpha={args.alpha}) | nodes={G.number_of_nodes()} edges={G.number_of_edges()}",
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(l=10, r=10, t=50, b=10),
            )

            html_path = out_dir / "graph.html"
            fig.write_html(str(html_path))
            info["graph_html"] = str(html_path)

    with open(out_dir / "info.json", "w") as f:
        json.dump({**info, "paths": {"edges": str(edges_path)}}, f, indent=2)

    print("Done. Wrote:", out_dir)
    print("Edges:", edges_path)
    if html_path:
        print("Graph:", html_path)
    else:
        print("Graph not written (install plotly + networkx to enable).")


if __name__ == "__main__":
    main()

