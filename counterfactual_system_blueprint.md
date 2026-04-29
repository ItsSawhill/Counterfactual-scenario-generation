# Counterfactual Generation System Blueprint

## Objective

Upgrade the current conditional diffusion workflow from:

- `future returns | 30-day asset window + 5 macro series`

to:

- `future cross-asset return distribution | live market state + release-aware macro state + explicit counterfactual shock + optional event/news context`

The existing DDPM remains the reference baseline. We improve the system in layers so every upgrade can be measured against the current model instead of replacing everything at once.

## Reference Baseline

Current system from the finished notebook:

- Targets: `SPY`, `QQQ`, `GLD`
- Macro conditioning: `FEDFUNDS`, `DGS10`, `CPIAUCSL`, `UNRATE`, `VIXCLS`
- Representation:
  - daily log returns for targets
  - daily aligned macro levels for conditioning
- Window length: `30`
- Model: conditional 1D DDPM with temporal residual blocks
- Baselines already used:
  - GBM / Monte Carlo
  - DDPM evaluation on statistical and financial diagnostics

This baseline should be preserved as `baseline_v0` for all later comparisons.

## Target Build Philosophy

The most important upgrade is not adding more model complexity first. The priority order is:

1. Fix timing realism in the data.
2. Expand the market state representation.
3. Introduce explicit shock semantics.
4. Add event/news context.
5. Improve the model architecture only after the data layer is stronger.

## Recommended Target Universe

### Preferred Core Universe

Use a compact macro cross-asset set:

- `SPY` or `ES=F`: broad US equity beta
- `QQQ` or `NQ=F`: growth / tech beta
- `GC=F`: gold futures
- `CL=F`: WTI crude oil futures
- `ZN=F`: 10-year Treasury note futures

### Optional Sixth Target

Choose one:

- `DX-Y.NYB`: dollar index
- `HYG`: high-yield credit ETF

### Guidance

- If the goal is macro / futures-style scenario generation, use a fully futures-oriented target set where possible.
- Keep `^VIX` or `VIXCLS` as a conditioning variable instead of a generated target in the first upgraded version.
- Do not jump to more than 5 or 6 targets initially. The state space expands quickly and makes evaluation harder.

## Conditioning Universe

### Core Macro Series

These should be treated as known-as-of-date observations, not hindsight-clean levels:

- `FEDFUNDS`
- `DGS2`
- `DGS10`
- `T10Y2Y`
- `CPIAUCSL`
- `UNRATE`
- `PAYEMS`
- `INDPRO`
- `VIXCLS`

### Optional Macro Series

Add only after the core set is stable:

- `PCEPI`
- `PPIACO`
- `RSAFS`
- `UMCSENT`
- `BAMLH0A0HYM2`
- `BAMLC0A0CM`

## Data Timing Rules

This is the highest-priority correction.

### Rule 1

Every row in the training data must reflect only information known on that calendar date.

### Rule 2

Macro series must be aligned by release availability, not by the observation month alone.

### Rule 3

If possible, use ALFRED vintages for macro data so the model learns from the value that was actually available at that time.

### Rule 4

Daily market series can be aligned on trading dates, but macro features must be forward-filled only after their release date.

### Rule 5

Scheduled event features should be timestamped separately from macro levels.

## Data Entities

### 1. Daily Market Panel

One row per trading date.

Columns:

- `Date`
- target prices
- target log returns
- target rolling realized vol
- target downside vol
- target rolling skew
- target rolling kurtosis
- target rolling drawdown
- target rolling volume shock if volume exists

### 2. Macro State Panel

One row per trading date with release-aware values.

Columns:

- `Date`
- macro latest available level
- macro delta vs previous release
- macro z-score vs trailing history
- macro 3m trend
- release-day indicator
- days-since-last-release

### 3. Event / News Panel

One row per event item.

Columns:

- `event_timestamp`
- `event_date`
- `event_type`
- `event_source`
- `entity`
- `topic`
- `scheduled_flag`
- `headline_text`
- `sentiment_score`
- `uncertainty_score`
- `hawkish_dovish_score`
- `shock_direction`
- `shock_magnitude_proxy`

### 4. Scenario Definition Table

Each counterfactual should be explicit and reproducible.

Columns:

- `scenario_id`
- `scenario_name`
- `series_name`
- `shock_mode`
- `shock_value`
- `shock_units`
- `shock_start_offset`
- `shock_duration_days`
- `notes`

## Feature Groups

Feature selection should happen at the group level first.

### Group A: Core Market State

- 1d log return for each target
- 5d cumulative return
- 20d cumulative return
- 5d realized vol
- 20d realized vol
- downside semivol
- rolling max drawdown

### Group B: Cross-Asset Structure

- gold minus equity return spread
- oil minus equity return spread
- growth minus broad equity spread
- 10y futures return or yield proxy
- dollar return if included
- rolling cross-asset correlation summaries

### Group C: Macro Levels And Changes

For each selected macro series:

- latest available level
- delta vs previous release
- trailing z-score
- 3m trend
- release-day flag
- days since release

### Group D: Regime And Volatility State

- `VIXCLS`
- realized / implied vol gap
- curve slope proxy
- credit spread proxy
- inflation regime flag
- growth regime flag
- policy regime flag

### Group E: Event / News Context

- event type embedding or one-hot bucket
- headline sentiment
- uncertainty score
- hawkish / dovish tone
- macro surprise direction
- days since last major event

### Group F: Counterfactual Shock Encoding

This should be separate from the raw observed state.

- shock target series
- shock type: `delta`, `percent`, `level`
- normalized shock magnitude
- shock persistence
- scenario family

## Recommended Feature Schema

### Targets

First upgraded target set:

- `SPY_log_return`
- `QQQ_log_return`
- `GC=F_log_return`
- `CL=F_log_return`
- `ZN=F_log_return`

If futures ticker symbols are inconvenient in file names, normalize them:

- `SPY_log_return`
- `QQQ_log_return`
- `GC_log_return`
- `CL_log_return`
- `ZN_log_return`

### Conditioning Columns

Minimum v1 conditioning schema:

- `FEDFUNDS_level`
- `FEDFUNDS_delta`
- `FEDFUNDS_z`
- `DGS10_level`
- `DGS10_delta`
- `DGS10_z`
- `T10Y2Y_level`
- `CPIAUCSL_level`
- `CPIAUCSL_delta`
- `CPIAUCSL_z`
- `UNRATE_level`
- `UNRATE_delta`
- `UNRATE_z`
- `PAYEMS_delta`
- `INDPRO_delta`
- `VIXCLS_level`
- `VIXCLS_z`
- `equity_realized_vol_20d`
- `commodity_realized_vol_20d`
- `rates_realized_vol_20d`
- `cross_asset_corr_20d`
- `release_day_flag`
- `days_since_fomc`
- `days_since_cpi`
- `days_since_nfp`

### Event / News Columns

Add in the next phase:

- `macro_event_count_3d`
- `fomc_tone_score`
- `inflation_news_sentiment_3d`
- `growth_news_sentiment_3d`
- `policy_uncertainty_score_3d`

## Sample Construction

### Window Design

Keep the current `30` day rolling sequence initially so the comparison against the existing DDPM stays clean.

Recommended sample structure:

- `X_target_history`: shape `[seq_len, num_targets]`
- `C_dynamic_features`: shape `[seq_len, num_condition_features]`
- `S_static_shock`: shape `[num_shock_features]`
- `Y_future_target_path`: generated by diffusion, same horizon as sequence length in v1

### Why Keep 30 First

- it matches the current trained system
- it reduces confounding during ablation
- it makes checkpoint comparison easier

After the first upgraded retrain, test:

- `30` day context / `30` day horizon
- `60` day context / `30` day horizon
- `60` day context / `60` day horizon

## Real-Time Counterfactual Semantics

Each generated scenario should be framed as:

- observed live market state
- observed live macro state
- explicit intervention
- generated distribution of future returns and prices

Example:

- baseline: no intervention
- hawkish policy shock: `FEDFUNDS +100 bps`, `DGS10 +50 bps`
- sticky inflation: `CPI level +1 percent`, `VIX +4`
- growth scare: `UNRATE +1`, `VIX +8`, `DGS10 -40 bps`

## Model Roadmap

### Model V0

Current DDPM, preserved unchanged.

### Model V1

Same DDPM architecture, but retrained on:

- release-aware macro data
- upgraded asset universe
- richer market-state features

This is the most important benchmark.

### Model V2

Add separate encoders for:

- target history
- dynamic macro / state features
- static shock vector

Fuse them before the denoising backbone.

### Model V3

Add analog retrieval:

- retrieve nearest historical windows by macro + market + vol state
- append analog summary features or embeddings
- condition diffusion on live state plus retrieved analog state

### Model V4

If regime behavior still looks blurred, test:

- regime gating
- mixture-of-experts
- classifier-free guidance for stronger shock conditioning

## Feature Selection Strategy

Do not start with single-column feature selection.

### Stage 1: Grouped Ablation

Run:

- baseline target history only
- `+ Group A`
- `+ Group A + B`
- `+ Group A + B + C`
- `+ Group A + B + C + D`
- `+ Group A + B + C + D + E`
- `+ Group A + B + C + D + E + F`

### Stage 2: Leave-One-Group-Out

After the best grouped model is identified, remove each group one at a time:

- remove cross-asset
- remove macro change features
- remove regime features
- remove event/news features
- remove shock vector

### Stage 3: Within-Group Pruning

Only then test smaller removals such as:

- remove kurtosis
- remove drawdown
- remove policy event features
- remove payroll features

## Evaluation Framework

Accuracy here should mean scenario realism and counterfactual usefulness, not just point prediction.

### Distribution Diagnostics

- mean / std error
- skew / kurtosis error
- Wasserstein distance
- KS statistic
- quantile error at `1%`, `5%`, `95%`, `99%`

### Time-Series Diagnostics

- ACF error
- volatility clustering replication
- drawdown behavior
- tail event frequency
- cross-asset correlation distance

### Counterfactual Diagnostics

These matter most for the final system:

- scenario monotonicity
  - does higher inflation shock worsen inflation-sensitive assets?
- sign consistency
  - do rates shocks affect duration-sensitive assets in the expected direction?
- tail calibration
  - do stress scenarios widen downside tails?
- scenario separation
  - are baseline and stress distributions meaningfully different?

### Event Window Backtests

Evaluate on real windows such as:

- CPI surprise days
- FOMC announcement days
- payroll days
- oil shock periods
- banking stress periods
- volatility spikes

## Experiment Matrix

### Data Experiments

- `D0`: current dataset and current DDPM
- `D1`: release-aware macro timing only
- `D2`: `D1 + upgraded target universe`
- `D3`: `D2 + market-state features`
- `D4`: `D3 + regime features`
- `D5`: `D4 + event/news features`

### Model Experiments

- `M0`: current DDPM
- `M1`: DDPM retrained on `D3`
- `M2`: DDPM with static shock encoder on `D3`
- `M3`: DDPM with shock encoder on `D4`
- `M4`: retrieval-augmented DDPM on `D4`
- `M5`: retrieval + event/news on `D5`

### Selection Rule

Do not advance because a model looks more sophisticated. Advance only if it improves:

- downside calibration
- cross-asset realism
- counterfactual monotonicity
- event-window performance

## Implementation Sequence

### Phase 1: Data Reliability

- freeze current baseline outputs
- define canonical target universe
- build release-aware macro loader
- build continuous futures loader for `GC`, `CL`, `ZN`
- generate aligned live-aware daily panel

### Phase 2: Richer State Features

- add realized vol, drawdown, rolling skew / kurtosis
- add cross-asset spread and correlation summaries
- add regime flags
- retrain current DDPM on this improved dataset

### Phase 3: Shock Semantics

- define scenario table
- separate observed state from intervention vector
- retrain with explicit shock encoding

### Phase 4: Event / News Layer

- ingest structured macro event calendar
- add timestamped headlines
- derive compact sentiment / uncertainty / tone features
- retrain and compare

### Phase 5: Retrieval

- build historical analog index
- retrieve nearest windows at inference time
- append analog features to conditioning

### Phase 6: Architecture Upgrade

- only after the best data configuration is clear
- test encoder fusion, guidance, and regime gating

## Suggested First Deliverables

Build these next:

1. `dataset_spec_v1.json`
2. `scenario_definitions.csv`
3. `release_aware_macro_loader.ipynb`
4. `continuous_futures_builder.ipynb`
5. `ablation_matrix.csv`

## Immediate Recommendation

The next concrete build step should be:

- create a new dataset spec and data loader that replaces `GLD` with `GC`, adds `CL` and `ZN`, and aligns macro data by release-aware timing while keeping the current DDPM architecture unchanged

That gives the cleanest answer to the question:

- "How much of the improvement comes from better data versus a different model?"
