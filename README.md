# Statistical Arbitrage via Market Segmentation

This repository implements a **statistical arbitrage research pipeline** that combines  
**market segmentation**, **dependence modeling**, and **mean-reversion trading signals**.

Rather than applying a single global model across all assets, the project first **segments the market into economically coherent clusters**, then constructs **relative-value signals within each segment**. The goal is to identify **stable, interpretable mean-reversion opportunities** that are masked at the aggregate market level.

---

## Project Overview

The pipeline follows four main stages:

1. **Market Segmentation**
   - Assets are grouped using both linear and nonlinear representations
   - Clusters are intended to reflect shared risk exposures rather than sectors alone

2. **Dependence Estimation**
   - Sparse precision matrices are estimated using **Graphical Lasso**
   - Emphasizes conditional dependence rather than raw correlation

3. **Signal Construction**
   - Mean-reversion signals are constructed using standardized residuals
   - Signals are normalized and clipped to control leverage and tail risk

4. **Backtesting and Evaluation**
   - Portfolio returns are constructed from cross-sectional signals
   - Performance metrics include cumulative returns, Sharpe ratios, and drawdowns

---

## Methods

### Market Segmentation

Two complementary approaches are used:

- **Graphical Lasso (GLASSO)**  
  Estimates sparse inverse covariance matrices to uncover conditional dependence
  structures between assets.

- **Autoencoders**  
  Nonlinear dimensionality reduction used to learn latent market structure and
  cluster assets in representation space.

These methods allow segmentation based on **shared dynamics**, not just correlations.

---

### Statistical Arbitrage Signals

Within each cluster:

- Asset returns are standardized using rolling statistics
- Z-scores are interpreted as relative mispricings
- Positions are taken long and short against the cluster mean

Signals are clipped to reduce sensitivity to extreme observations and to stabilize
portfolio risk.

---

### Backtesting Framework

The backtesting engine supports:

- Rolling estimation windows
- Cluster-level neutrality
- Equal-weighted or volatility-scaled portfolios
- Transaction-cost-free baseline evaluation

Results are stored in a structured `results/` directory for reproducibility.

---
## Presentation:
[Final Presentation (PDF)](report/Stat Arb Final Presentation.pdf)
