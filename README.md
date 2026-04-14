# UrbanFlow: Hierarchical Marketplace Forecasting & Causal Elasticity Engine

## 📌 Executive Summary
UrbanFlow is a production-ready data pipeline and machine learning engine designed to forecast high-volume marketplace demand and measure pricing elasticity. 

Using over 15 million raw NYC Taxi & Limousine Commission (TLC) records, this project establishes a scalable data architecture to predict zone-level hourly demand and utilizes econometric causal inference to measure the exact demand shock of a real-world regulatory price hike.

### **Key Business Outcomes**
* **Forecasting Accuracy:** Engineered a gradient-boosted engine (LightGBM) with temporal rolling lags that achieved an **18.09% WAPE**, outperforming standard statistical baselines (ARIMA/Seasonal Naive) by a relative ~27%.

**Baseline: 24.75%**

**LightGBM: 18.09% (Real-time speed)**

**Chronos-T5 (AI): 11.10% (Maximum accuracy)**


* **Pricing Elasticity (Causal Inference):** Deployed a Difference-in-Differences (DiD) econometric model to isolate the impact of the February 2019 $2.50 Congestion Surcharge, proving with statistical significance how the targeted price hike impacted Manhattan demand relative to a control group.

---

## 🏗️ Technical Architecture

This project is built to mimic a modern, serverless cloud environment running locally:

1. **Data Engineering (The Data Layer)**
   * **Stack:** `DuckDB`, `SQL`, `Parquet`
   * **Action:** Ingests and aggregates gigabytes of raw Parquet files via in-process SQL execution, transforming event-level transaction logs into clean, hierarchical time-series tables without loading raw data into memory.
2. **Predictive Modeling (The Forecasting Engine)**
   * **Stack:** `LightGBM`, `MLForecast` (Nixtla), `StatsForecast`
   * **Action:** Constructs baseline models (AutoARIMA) and advanced hybrid models leveraging exogenous regressors (hour of day, day of week) and autoregressive lags to predict highly seasonal marketplace demand.
3. **Econometrics (The Causal Layer)**
   * **Stack:** `Statsmodels`, `SciPy`, `Seaborn`
   * **Action:** Bypasses A/B testing by using historical natural experiments. Computes the price elasticity of demand using Ordinary Least Squares (OLS) regression on log-transformed usage data.

---

## 📂 Project Structure

```text
urbanflow-engine/
│
├── 01_build_data_layer.py       # DuckDB pipeline to aggregate Jan 2019 Parquet logs
├── 02_baseline_forecast.py      # ARIMA & Seasonal Naive baseline benchmarking
├── 03_advanced_forecast.py      # LightGBM engine with lag & temporal feature engineering
├── 04_causal_data_prep.py       # Joins Feb 2019 data with geographic TLC zone maps
├── 05_causal_analysis.py        # Difference-in-Differences regression and elasticity output
├── causal_impact_dashboard.png  # Executive visualization of the demand shock
└── README.md                    # Project documentation