# Methodology

## Problem Framing

The project models future cross-asset return paths under macro-conditioned and shock-conditioned states. Instead of forecasting a single point estimate, it generates scenario distributions that can be inspected for:

- return dispersion
- tail losses
- cross-asset co-movement
- sensitivity to macro regime assumptions

## Data Construction

The notebook pipeline assembles three broad data views:

### 1. Market state

Daily market features are built for a five-asset target universe:

- `SPY`
- `QQQ`
- `GC`
- `CL`
- `ZN`

Feature examples include:

- log returns
- rolling returns
- realized volatility
- downside semivolatility
- rolling skew and kurtosis
- drawdown and volume shock features where available

### 2. Macro state

Macro conditioning series are built with release-aware alignment rather than hindsight-perfect monthly joins. The tracked configuration shows features for:

- `FEDFUNDS`
- `DGS2`
- `DGS10`
- `T10Y2Y`
- `CPIAUCSL`
- `UNRATE`
- `PAYEMS`
- `INDPRO`
- `VIXCLS`

Each series can contribute level, change, and normalized state features.

### 3. Cross-asset and event timing context

The aligned panel also includes engineered spread and state features such as:

- gold minus equity return spread
- oil minus equity return spread
- growth minus broad equity return spread
- rates minus equity return spread
- cross-asset correlation summaries
- release timing flags like `days_since_cpi`, `days_since_fomc`, and `days_since_nfp`

## Windowing And Conditioning

The current upgraded configuration uses:

- `seq_len = 30`
- `input_dim = 5`
- `condition_dim = 88`

Training data is split into train, validation, and test windows, then scaled before being passed into the model.

## Model

The trained model is a conditional 1D DDPM with:

- hidden dimension `96`
- time embedding dimension `192`
- `8` temporal residual blocks
- diffusion step count `300`
- dropout `0.05`

This keeps the project anchored in generative time-series modeling rather than a classical Monte Carlo-only workflow.

## Evaluation Method

The upgraded DDPM is compared against a Gaussian baseline using:

- marginal distribution fit
- tail quantile accuracy
- Wasserstein distance
- lag-1 autocorrelation diagnostics
- cross-asset correlation structure
- scenario-level risk metrics such as downside probability, VaR, and CVaR

## Scenario Lab

The repository also includes a React frontend that frames the generated scenarios as a decision-support surface. It consumes simulation outputs and presents:

- scenario path fans
- distribution comparisons
- risk metrics
- portfolio-oriented scenario interpretation

The UI is useful for demos and recruiter-facing walkthroughs, even though the corresponding backend API is not fully packaged in this repository.
