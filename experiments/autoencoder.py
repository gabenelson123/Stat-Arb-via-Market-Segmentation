#!/usr/bin/env python3
"""
Autoencoder vs PCA reconstruction on standardized returns.

Steps (matches your notebook logic):
- load Bloomberg-wide prices
- active universe filtering
- log returns
- RobustScaler -> StandardScaler
- train Autoencoder
- PCA baseline with same latent dim
- save comparison plot + metrics JSON

Outputs:
- cleaned_returns.csv (optional)
- pca_vs_autoencoder.png
- metrics.json
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

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler

# Torch is required for this experiment
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        width1 = min(2048, max(512, input_dim // 2))
        width2 = min(1024, max(256, input_dim // 4))
        act = nn.GELU

        self.enc = nn.Sequential(
            nn.Linear(input_dim, width1),
            nn.BatchNorm1d(width1),
            act(),
            nn.Linear(width1, width2),
            nn.BatchNorm1d(width2),
            act(),
            nn.Linear(width2, latent_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, width2),
            nn.BatchNorm1d(width2),
            act(),
            nn.Linear(width2, width1),
            nn.BatchNorm1d(width1),
            act(),
            nn.Linear(width1, input_dim),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)
        return out

    def encode(self, x):
        return self.enc(x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="results/autoencoder")
    p.add_argument("--window", type=int, default=252)
    p.add_argument("--universe-n", type=int, default=1200)
    p.add_argument("--winsor-q", type=float, default=0.005)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--smooth-l1-weight", type=float, default=0.5)
    p.add_argument("--save-cleaned-csv", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading prices...")
    prices = load_bloomberg_wide_pxlast(args.data)
    prices = filter_active_columns(prices, window=args.window, drop_name_substr="Index")
    if args.universe_n and args.universe_n > 0:
        prices = prices.iloc[:, : min(args.universe_n, prices.shape[1])].copy()

    rets = numpy.log(prices).diff()
    rets = winsorize_returns(rets, args.winsor_q)
    rets = rets.dropna(axis=0, how="any")  # AE expects a complete matrix

    if rets.shape[0] < 20 or rets.shape[1] < 20:
        raise RuntimeError(f"Not enough complete data after cleaning: rets shape={rets.shape}")

    # Standardize (your pattern: Robust then Standard)
    X_raw = rets.values.astype(numpy.float32)
    robust = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))
    X_r = robust.fit_transform(X_raw)

    std = StandardScaler(with_mean=False, with_std=True)
    X = std.fit_transform(X_r).astype(numpy.float32)

    if args.save_cleaned_csv:
        cleaned_path = out_dir / "cleaned_returns.csv"
        pandas.DataFrame(X, index=rets.index, columns=rets.columns).to_csv(cleaned_path)
        print("Saved cleaned returns:", cleaned_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    X_tensor = torch.from_numpy(X)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = Autoencoder(input_dim=X.shape[1], latent_dim=args.latent_dim).to(device)

    mse = nn.MSELoss()
    huber = nn.SmoothL1Loss(beta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    losses = []
    model.train()
    for epoch in range(args.epochs):
        tot = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(xb)
            loss_mse = mse(recon, xb)
            loss_huber = huber(recon, xb)
            loss = (1 - args.smooth_l1_weight) * loss_mse + args.smooth_l1_weight * loss_huber
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tot += float(loss.item())
        avg = tot / max(1, len(loader))
        losses.append(avg)
        scheduler.step()
        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss {avg:.6f}")

    # Reconstructions
    model.eval()
    with torch.no_grad():
        X_ae = model(X_tensor.to(device)).cpu().numpy()

    pca = PCA(n_components=args.latent_dim, random_state=42)
    Zp = pca.fit_transform(X)
    X_pca = pca.inverse_transform(Zp)

    ae_mse = float(mean_squared_error(X, X_ae))
    pca_mse = float(mean_squared_error(X, X_pca))
    pca_expl = float(pca.explained_variance_ratio_.sum())

    # Plot (simplified but same spirit as your notebook figure)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(losses)
    ax.set_title("Autoencoder Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_png = out_dir / "loss_curve.png"
    plt.savefig(loss_png, dpi=200, bbox_inches="tight")
    plt.close()

    # Error distributions
    ae_err = numpy.mean((X - X_ae) ** 2, axis=1)
    pca_err = numpy.mean((X - X_pca) ** 2, axis=1)
    plt.figure(figsize=(9, 5.5))
    plt.hist(ae_err, bins=50, alpha=0.7, label="Autoencoder", density=True)
    plt.hist(pca_err, bins=50, alpha=0.7, label="PCA", density=True)
    plt.title(f"Per-sample reconstruction error (latent_dim={args.latent_dim})")
    plt.xlabel("MSE per sample")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    comp_png = out_dir / "pca_vs_autoencoder.png"
    plt.savefig(comp_png, dpi=200, bbox_inches="tight")
    plt.close()

    metrics = {
        "data": args.data,
        "window": args.window,
        "winsor_q": args.winsor_q,
        "returns_shape": [int(rets.shape[0]), int(rets.shape[1])],
        "latent_dim": int(args.latent_dim),
        "epochs": int(args.epochs),
        "ae_mse": ae_mse,
        "pca_mse": pca_mse,
        "pca_explained_variance": pca_expl,
        "improvement_pct": float((pca_mse - ae_mse) / max(1e-12, pca_mse) * 100.0),
        "paths": {"loss_curve": str(loss_png), "comparison": str(comp_png)},
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done. Wrote:", out_dir)
    print("AE MSE:", ae_mse, "| PCA MSE:", pca_mse, "| improvement %:", metrics["improvement_pct"])


if __name__ == "__main__":
    main()

