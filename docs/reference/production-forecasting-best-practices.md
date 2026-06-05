# Production-Grade Time Series Forecasting: Best Practices Reference

- **Date**: 2026-04-05 20:23
- **Status**: completed
- **Module**: system-wide reference

---

## 1. Evaluation Metrics

### What & Why

Point-forecast accuracy metrics quantify how close predictions are to actual values. For financial time series, choosing the right metric is critical because different metrics penalize errors differently and have different failure modes (e.g., MAPE is undefined when actual values are zero). A mature forecasting service should report multiple complementary metrics rather than relying on a single number.

### Key Metrics

| Metric | Formula Intuition | Strengths | Weaknesses |
|--------|-------------------|-----------|------------|
| **MAE** | Mean of absolute errors | Interpretable in original units; robust to outliers | Scale-dependent; cannot compare across series |
| **RMSE** | Root of mean squared errors | Penalizes large errors more heavily | Scale-dependent; sensitive to outliers |
| **MAPE** | Mean of |error/actual| as % | Scale-independent; intuitive percentage | Undefined when actual=0; asymmetric (penalizes over-forecasts more) |
| **sMAPE** | Symmetric MAPE variant | Bounded [0%, 200%] | Still has mathematical issues; Hyndman & Koehler (2006) recommend against it |
| **MASE** | MAE scaled by naive forecast MAE | Scale-independent; well-defined for zero values; directly interpretable (< 1 means better than naive) | Requires a training set for scaling |
| **WAPE/WMAPE** | Weighted MAPE (total error / total actual) | Handles zeros gracefully; good for aggregated accuracy | Can mask poor performance on small-value items |
| **MDA** (Mean Directional Accuracy) | % of times direction of change is correct | Critical for trading strategies where direction matters more than magnitude | Ignores magnitude of errors entirely |
| **CRPS** (Continuous Ranked Probability Score) | Evaluates full predictive distribution | Gold standard for probabilistic forecasts | More complex to compute and interpret |

### Practical Recommendations

- Always report **MASE** as the primary metric -- it directly tells you whether your model beats the naive baseline (MASE < 1.0).
- For financial applications, add **MDA** (directional accuracy) since correct direction can be more valuable than precise magnitude.
- For probabilistic forecasts, use **CRPS** or **quantile loss** to evaluate the full prediction distribution, not just point estimates.
- Report metrics at multiple horizons (1-step, 5-step, 10-step) since accuracy typically degrades with longer horizons.

### Authoritative Sources

1. [Hyndman & Athanasopoulos -- *Forecasting: Principles and Practice* (3rd ed), Chapter 5.8: Evaluating Point Forecast Accuracy](https://otexts.com/fpp3/accuracy.html) -- The definitive textbook reference for forecast accuracy metrics, including the original MASE proposal.
2. [AutoGluon -- Forecasting Time Series: Evaluation Metrics](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-metrics.html) -- Practical documentation covering when to use each metric with code examples, from Amazon's AutoML framework.
3. [Nixtla -- Evaluation Metrics Documentation](https://www.nixtla.io/docs/forecasting/evaluation/evaluation_metrics) -- Concise reference covering MAE, RMSE, MAPE, MASE, and CRPS with their use cases.

---

## 2. Backtesting / Walk-Forward Validation

### What & Why

Standard k-fold cross-validation is **invalid for time series** because it breaks temporal ordering, allowing the model to train on future data and evaluate on past data ("data leakage"). Walk-forward validation (also called time series cross-validation) respects chronological order: train on the past, test on the immediate future, roll forward, repeat.

### Methods

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Expanding Window** | Training set grows with each fold; test window rolls forward | Default choice; leverages all available history |
| **Rolling/Sliding Window** | Training set has fixed size; both train and test windows slide forward | When older data may be less relevant (concept drift) |
| **Purged/Embargoed CV** | Expanding window + gap between train and test sets | Financial data where autocorrelation creates subtle leakage |

### Common Leakage Pitfalls

- **Normalization before splitting**: Computing mean/std over the full dataset, then splitting. Always fit scalers on training data only.
- **Feature engineering before splitting**: Applying moving averages, decomposition, or technical indicators over the full series before partitioning.
- **Backfilling missing values**: Using future values to fill gaps in the past. Only forward-fill is safe.
- **Shuffling**: Any random shuffling of time series data before splitting destroys temporal ordering.

### Practical Recommendations

- Use scikit-learn's `TimeSeriesSplit` with the `gap` parameter to introduce an embargo period between train and test sets.
- For financial time series, consider **Combinatorial Purged Cross-Validation** (de Prado, 2018) which addresses false discovery rates in walk-forward analysis.
- Always refit the model at each fold in production-like backtesting; "no-refit" backtesting is faster but less realistic.
- Use libraries like **skforecast** that provide built-in backtesting with both expanding and rolling window strategies.

### Authoritative Sources

1. [scikit-learn -- TimeSeriesSplit Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) -- Official implementation with `gap` parameter for embargo periods.
2. [Hewamalage et al. (2023) -- "Forecast evaluation for data scientists: common pitfalls and best practices" (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9718476/) -- Peer-reviewed paper covering evaluation pitfalls specific to time series, including leakage taxonomy.
3. [skforecast -- Backtesting Forecaster Documentation](https://skforecast.org/0.5.0/user_guides/backtesting) -- Practical guide implementing walk-forward validation with scikit-learn compatible models.

---

## 3. Baseline Models

### What & Why

A baseline model is the simplest reasonable forecast for your problem. It sets the **minimum bar** that any more complex model must beat to justify its additional complexity, compute cost, and maintenance burden. Without baselines, you cannot answer: "Is my deep learning model actually good, or would a last-value repeat do just as well?"

### Standard Baselines

| Baseline | Method | Best For |
|----------|--------|----------|
| **Naive (Persistence)** | Forecast = last observed value | Non-seasonal, random-walk-like data (e.g., stock prices) |
| **Seasonal Naive** | Forecast = value from same season last cycle | Data with strong seasonality |
| **Drift** | Naive + linear trend extrapolation | Data with clear upward/downward trend |
| **Moving/Window Average** | Average of last k observations | Smoothing noisy data; local trend capture |
| **Historical Average** | Mean of all historical values | Stationary data |
| **ARIMA/SARIMA** | Statistical autoregressive model | Structured statistical baseline before ML |

### Why Baselines Matter in Practice

- Recent research (Springer, 2025) warns that studies comparing only complex models against each other -- without naive baselines -- produce "limited conclusions and reduced practical significance."
- In financial time series, the naive forecast (random walk) is notoriously hard to beat. Proving your model beats it is the first real test of value.
- Baselines establish a lower bound for acceptable production performance. If a deployed model degrades to baseline-level accuracy, that is a clear retraining trigger.

### Practical Recommendations

- Always implement at least **Naive** and **Seasonal Naive** baselines before building complex models.
- Use **MASE** as your primary metric since it directly compares against the naive baseline (MASE = 1.0 means you are exactly as good as naive).
- For currency exchange rates, the naive (random walk) forecast is an especially strong baseline -- your model must demonstrably beat it.
- Use Nixtla's **StatsForecast** library for fast, production-quality baseline implementations.

### Authoritative Sources

1. [Nixtla -- "Unlocking the Power of Baseline Forecasts" (Blog)](https://www.nixtla.io/blog/baseline-forecasts) -- Why baselines matter, with code examples using StatsForecast.
2. [Nixtla StatsForecast -- Models Documentation](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html) -- Reference for Naive, SeasonalNaive, WindowAverage, HistoricAverage, and RandomWalkWithDrift implementations.
3. [Springer (2025) -- "Mind the naive forecast! A rigorous evaluation of forecasting models for time series with low predictability"](https://link.springer.com/article/10.1007/s10489-025-06268-w) -- Recent peer-reviewed paper demonstrating the importance of naive baselines.

---

## 4. Model Monitoring & Drift Detection

### What & Why

Once deployed, a model's performance will degrade over time as the real-world data distribution shifts. In financial markets, regime changes (policy shifts, crises, structural breaks) can invalidate model assumptions rapidly. **Monitoring** detects this degradation before it causes costly errors; **drift detection** identifies the root cause.

### Types of Drift

| Type | Definition | Detection Method |
|------|------------|------------------|
| **Data Drift** (covariate shift) | Input feature distributions change (e.g., a currency pair's volatility regime shifts) | Compare input distributions: KS test, PSI, Wasserstein distance |
| **Concept Drift** | The relationship between inputs and outputs changes (e.g., the model's learned patterns no longer hold) | Monitor prediction error metrics over time; Page-Hinkley test, ADWIN |
| **Label Drift** | Target variable distribution changes | Compare target distributions between windows |

### Detection Methods

- **Statistical tests**: Kolmogorov-Smirnov (KS) test, Chi-squared test for categorical features, Population Stability Index (PSI).
- **Distance metrics**: Wasserstein distance, KL divergence, Jensen-Shannon divergence.
- **Performance monitoring**: Track MAE/MASE on recent predictions vs. historical performance using rolling windows.
- **ADWIN (Adaptive Windowing)**: Automatically detects change points in streaming data.

### Practical Recommendations

- Monitor **both** input distributions (data drift) and output accuracy (concept drift) -- they often co-occur but not always.
- Set up alerting thresholds: e.g., if rolling MASE exceeds 1.0 (worse than naive), trigger investigation.
- For financial time series, monitor volatility regimes -- a shift from low to high volatility often signals concept drift.
- Use **Evidently AI** (open source) for automated drift reports with 20+ built-in statistical tests.
- Research shows adaptive retraining (triggered by drift detection) outperforms periodic retraining by ~9.3% in accuracy.

### Authoritative Sources

1. [Evidently AI -- "What is data drift in ML, and how to detect and handle it"](https://www.evidentlyai.com/ml-in-production/data-drift) -- Comprehensive guide on data drift detection methods with practical examples.
2. [Evidently AI -- "What is concept drift in ML, and how to detect and address it"](https://www.evidentlyai.com/ml-in-production/concept-drift) -- Concept drift taxonomy and detection strategies.
3. [Frontiers in AI (2024) -- "One or two things we know about concept drift -- a survey on monitoring in evolving environments"](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1330257/full) -- Comprehensive peer-reviewed survey on drift detection methods.

---

## 5. Prediction Intervals / Uncertainty Quantification

### What & Why

Point forecasts alone are insufficient for decision-making. A prediction of "USD/TWD = 30.5 tomorrow" is far less useful than "USD/TWD = 30.5 +/- 0.3 with 90% confidence." Uncertainty quantification enables risk management, position sizing, and informed decision-making.

### Methods

| Method | Description | Properties |
|--------|-------------|------------|
| **Quantile Regression** | Directly predicts quantiles (e.g., 5th and 95th percentile) | Flexible; can model asymmetric intervals; no distributional assumptions |
| **Conformal Prediction** | Distribution-free method providing guaranteed coverage | Finite-sample coverage guarantees; model-agnostic |
| **Conformalized Quantile Regression (CQR)** | Combines quantile regression + conformal prediction | Adaptive intervals (vary with input) + coverage guarantees |
| **Bootstrap** | Resample residuals to estimate prediction distribution | Simple to implement; can be computationally expensive |
| **Bayesian Methods** | Full posterior predictive distribution | Natural uncertainty quantification; can be computationally expensive |

### Conformal Prediction for Time Series

Standard conformal prediction assumes exchangeable (i.i.d.) data, which time series violates. Recent research addresses this:

- **KOWCPI** (Kernel-based Optimally Weighted Conformal Prediction Intervals): Adapts conformal methods for dependent data with data-adaptive weights, producing narrower intervals without sacrificing coverage.
- **CoRel** (Conformal Relational Prediction): Exploits correlations between multiple time series for tighter intervals.
- **Width-Adaptive Conformal Inference (WACI)**: Allows interval widths to vary with predictive uncertainty while maintaining theoretical coverage guarantees.

### Practical Recommendations

- Use **MAPIE** (scikit-learn-contrib) for conformal prediction intervals -- it integrates seamlessly with scikit-learn pipelines and supports time series.
- Start with **Conformalized Quantile Regression (CQR)** as the default method: it provides both adaptive intervals and coverage guarantees.
- Always evaluate calibration: check that your 90% prediction intervals actually contain 90% of observations.
- For financial applications, prediction intervals enable proper risk assessment and position sizing.

### Authoritative Sources

1. [MAPIE -- Model Agnostic Prediction Interval Estimator Documentation](https://mapie.readthedocs.io/) -- Official docs for the scikit-learn-compatible conformal prediction library.
2. [Romano et al. -- "Conformalized Quantile Regression" (NeurIPS 2019)](https://papers.neurips.cc/paper/8613-conformalized-quantile-regression.pdf) -- The foundational paper combining quantile regression with conformal prediction.
3. [Angelopoulos & Bates (2025) -- "Conformal Prediction: A Data Perspective" (ACM Computing Surveys)](https://dl.acm.org/doi/10.1145/3736575) -- Comprehensive recent survey covering conformal prediction theory and practice.

---

## 6. Feature Engineering for Time Series

### What & Why

Feature engineering transforms raw time series into informative inputs for ML models. For financial time series, this includes technical indicators, calendar features, and lagged variables. The critical constraint is **avoiding look-ahead bias** -- accidentally using future information during feature computation.

### Feature Categories

| Category | Examples | Pitfalls |
|----------|----------|----------|
| **Lag features** | y(t-1), y(t-2), ..., y(t-k) | Ensure lags are sufficient; avoid lag=0 (look-ahead) |
| **Rolling statistics** | Moving average, rolling std, rolling min/max | Window must only look backward; use `.shift(1)` before `.rolling()` |
| **Technical indicators** | RSI, MACD, Bollinger Bands, ATR | Compute on training data only; some indicators use future data by default |
| **Calendar features** | Day of week, month, quarter, holiday flags | Generally safe; no temporal leakage |
| **Fourier features** | Sine/cosine terms at seasonal frequencies | Safe if frequency is predetermined; do not fit frequency on test data |
| **Cross-series features** | Correlations with other currency pairs, market indices | Must use lagged values of external series |

### Look-Ahead Bias -- Critical Rules

1. **Never compute features on the full dataset before train/test splitting.** Moving averages, normalization, and decomposition must be computed within the training window only.
2. **Shift before rolling.** When creating rolling features like `rolling_mean_5 = y.shift(1).rolling(5).mean()`, the `.shift(1)` ensures you do not include the current observation.
3. **Forward-fill only.** When imputing missing values, never backfill -- it uses future information.
4. **Normalize per fold.** Fit scalers (StandardScaler, MinMaxScaler) on training data; transform test data using training statistics.

### Practical Recommendations

- Use scikit-learn's `TimeSeriesSplit` + pipeline to ensure transformations are fit only on training data.
- For currency forecasting, useful features include: lagged returns (not prices), volatility measures (rolling std of returns), and momentum indicators (RSI, MACD).
- Use feature selection (e.g., `feature_importances_` from tree models) to prune uninformative features and reduce overfitting.
- Consider using **feature-engine's** `LagFeatures` and `WindowFeatures` transformers which handle temporal ordering correctly.

### Authoritative Sources

1. [scikit-learn -- "Lagged features for time series forecasting" Tutorial](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) -- Official scikit-learn example demonstrating proper lag feature construction with `TimeSeriesSplit`.
2. [feature-engine -- Forecasting Features Documentation](https://feature-engine.trainindata.com/en/1.8.x/user_guide/timeseries/forecasting/index.html) -- Library providing sklearn-compatible transformers for lag features, window features, and datetime features with proper temporal handling.
3. [Microsoft -- "Introduction to feature engineering for time series forecasting"](https://medium.com/data-science-at-microsoft/introduction-to-feature-engineering-for-time-series-forecasting-620aa55fcab0) -- Microsoft's guide covering time series feature engineering best practices.

---

## 7. Retraining Strategy

### What & Why

Models degrade over time as data distributions shift. A retraining strategy defines **when** and **how** to update the model. Too frequent retraining wastes compute; too infrequent retraining allows performance to degrade.

### Strategies

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Periodic** | Retrain on a fixed schedule (daily, weekly, monthly) | Simple; predictable compute costs | May retrain when unnecessary or too late |
| **Trigger-based** | Retrain when drift is detected or accuracy drops below threshold | Responsive to actual degradation | Requires robust monitoring infrastructure |
| **Adaptive** | Combine periodic + trigger-based | Best accuracy improvement (~9.3% over baseline) | Most complex to implement |
| **Online/Incremental** | Continuously update model weights with new data | No retraining downtime; adapts quickly | Not all models support it; risk of catastrophic forgetting |

### Recent Research Findings (2025)

A 2025 paper on global forecasting models ("On the retraining frequency of global forecasting models", arXiv) found that:
- Less frequent retraining **does not harm** (and sometimes improves) forecast accuracy.
- Periodic retraining offers a good balance between predictive performance and computational efficiency.
- This challenges the conventional belief that frequent retraining is essential.

### Practical Recommendations

- Start with **weekly periodic retraining** for currency forecasting -- daily is likely excessive based on recent research.
- Implement **trigger-based guards**: if rolling MASE exceeds a threshold (e.g., 1.2), trigger immediate retraining regardless of schedule.
- When retraining, use a **rolling window** of recent data (e.g., last 2 years) rather than all historical data -- this naturally handles concept drift.
- Always keep the previous model version available for rollback.
- Log every retraining event: timestamp, data window used, metrics before/after, model version.

### Authoritative Sources

1. [arXiv (2025) -- "On the retraining frequency of global forecasting models"](https://arxiv.org/html/2505.00356v2) -- Recent research challenging frequent retraining assumptions, with experiments on retail demand forecasting.
2. [ML in Production -- "The Ultimate Guide to Model Retraining"](https://mlinproduction.com/model-retraining/) -- Comprehensive guide covering periodic, trigger-based, and online retraining approaches.
3. [SmartDev -- "AI Model Drift & Retraining: A Guide for ML System Maintenance"](https://smartdev.com/ai-model-drift-retraining-a-guide-for-ml-system-maintenance/) -- Practical guide connecting drift detection to retraining decisions.

---

## 8. MLOps for Time Series

### What & Why

MLOps (Machine Learning Operations) provides the infrastructure for reproducible, reliable, and maintainable ML systems. For time series forecasting, MLOps has additional requirements: temporal data versioning, walk-forward validation pipelines, and model monitoring that respects time ordering.

### Core Components

| Component | Purpose | Tools |
|-----------|---------|-------|
| **Experiment Tracking** | Log hyperparameters, metrics, artifacts for every training run | MLflow, Weights & Biases (W&B) |
| **Model Registry** | Version and stage models (staging -> production -> archived) | MLflow Model Registry, DVC |
| **Data Versioning** | Track which data was used for each model version | DVC, LakeFS |
| **Pipeline Orchestration** | Automate collect -> train -> evaluate -> deploy workflows | Airflow, Prefect, Metaflow |
| **Model Monitoring** | Track prediction accuracy and drift in production | Evidently AI, Grafana |
| **Reproducibility** | Recreate any historical model exactly | Git + DVC + MLflow + Docker |

### Time Series-Specific MLOps Concerns

- **Temporal data versioning**: You need to track not just "which dataset" but "which time window" was used for training. A model trained on 2023-01-01 to 2025-12-31 is fundamentally different from one trained on 2024-01-01 to 2025-12-31.
- **Walk-forward validation in CI/CD**: Automated pipelines should run walk-forward backtesting, not random splits, as part of model validation before deployment.
- **Prediction logging**: Store every prediction with its timestamp, input features, and model version for later evaluation against actuals.
- **Champion/Challenger**: Run the new model in shadow mode alongside the current production model before promoting it.

### Practical Recommendations

- Use **MLflow** for experiment tracking and model registry -- it is the most widely adopted open-source platform and integrates with scikit-learn, PyTorch, and TensorFlow.
- Version data with **DVC** (Data Version Control) alongside code in Git -- this ensures full reproducibility.
- Implement a **prediction logging** table: `(timestamp, model_version, input_features, prediction, actual, error)`. This enables retrospective analysis and drift detection.
- Containerize the training and inference pipeline (Docker) so that the environment is fully reproducible.
- For this project specifically: the existing Docker Compose setup is a strong foundation; adding MLflow tracking and DVC data versioning would be the highest-impact next steps.

### Authoritative Sources

1. [MLflow Documentation -- Model Versioning with MLflow](https://www.javacodegeeks.com/2025/06/model-versioning-with-mlflow-tracking-and-managing-your-ml-models.html) -- Practical guide to MLflow's tracking and model versioning capabilities.
2. [Google Research -- "A decoder-only foundation model for time-series forecasting" (TimesFM)](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/) -- Google's approach to production time series forecasting at scale, including their deployment methodology.
3. [Uber Engineering -- "Introducing Orbit, An Open Source Package for Time Series Inference and Forecasting"](https://www.uber.com/blog/orbit/) -- How Uber built and deployed production time series forecasting across fraud detection, capacity planning, and marketing budget allocation.

---

## Summary: Priority Recommendations for This Project

Based on the current architecture (PatchTST models, scikit-learn/PyTorch, Docker Compose), here is a prioritized list of improvements:

### High Priority (Foundation)

1. **Add baseline models** (Naive, Seasonal Naive) and always report MASE alongside existing metrics.
2. **Implement walk-forward backtesting** using `TimeSeriesSplit` with an embargo gap instead of simple train/test splits.
3. **Add prediction intervals** using MAPIE/conformal prediction to quantify forecast uncertainty.

### Medium Priority (Quality)

4. **Add directional accuracy (MDA)** as a metric -- critical for currency trading decisions.
5. **Audit feature engineering** for look-ahead bias: ensure all rolling features use `.shift(1)`, normalization is per-fold, and no future data leaks.
6. **Implement drift monitoring** with Evidently AI -- track input distribution shifts and prediction accuracy over time.

### Lower Priority (Scale)

7. **Add MLflow experiment tracking** to log training runs, hyperparameters, and metrics.
8. **Define retraining strategy**: weekly periodic retraining with trigger-based guards when MASE degrades.
9. **Add DVC** for data versioning to ensure full reproducibility.
