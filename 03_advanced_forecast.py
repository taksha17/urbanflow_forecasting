import duckdb
import pandas as pd
import time
from mlforecast import MLForecast
from lightgbm import LGBMRegressor

print("Loading data from DuckDB...")
con = duckdb.connect('urbanflow.db')

# --- Step 1. Data Extraction ---
query = """
WITH TopZones AS (
    SELECT pickup_zone_id, SUM(total_trips) as grand_total
    FROM hourly_zone_demand
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 20
)
SELECT 
    CAST(h.pickup_zone_id AS VARCHAR) AS unique_id,
    h.pickup_hour AS ds,
    h.total_trips AS y
FROM hourly_zone_demand h
JOIN TopZones t ON h.pickup_zone_id = t.pickup_zone_id
ORDER BY unique_id, ds;
"""

df = con.execute(query).df()
con.close()

# --- NEW: Data Imputation (Fixing the Missing Timestamps) ---
print("Cleaning data: Imputing missing hours with 0 trips...")

# Create a complete grid of every hour for every zone
all_zones = df['unique_id'].unique()
all_hours = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='h')

# Create a MultiIndex of all possible combinations
full_idx = pd.MultiIndex.from_product([all_zones, all_hours], names=['unique_id', 'ds'])

# Reindex our dataframe to this full grid, filling missing hours with 0
df = df.set_index(['unique_id', 'ds']).reindex(full_idx, fill_value=0).reset_index()


# --- Step 2. Train/Test Split (Fixing the Pandas Warning) ---
horizon = 168 
df = df.sort_values(['unique_id', 'ds'])

# A much cleaner way to split time-series that avoids the Pandas apply() warning
test_df = df.groupby('unique_id').tail(horizon)
train_df = df.drop(test_df.index)

print(f"Training on {len(train_df)} rows. Testing on {len(test_df)} rows.")


# --- Step 3. Define the ML Engine ---
mlf = MLForecast(
    models=[LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)],
    freq='h',
    lags=[1, 24, 168], # Look back 1 hour, 1 day, and 1 week ago
    date_features=['hour', 'dayofweek'] # Extract exogenous time features
)


# --- Step 4. Train and Forecast ---
print("Training LightGBM Engine. Watch how fast this is...")
start_time = time.time()

# Fit the model and predict
mlf.fit(train_df)
forecast_df = mlf.predict(horizon)

execution_time = time.time() - start_time
print(f"ML Forecasting completed in {execution_time:.2f} seconds.")


# --- Step 5. Accuracy Tracking ---
results_df = forecast_df.merge(test_df, on=['unique_id', 'ds'], how='left')

def calculate_wape(y_true, y_pred):
    # Safety check to avoid division by zero
    if y_true.sum() == 0: return 0
    return (abs(y_true - y_pred).sum() / y_true.sum()) * 100

print("\n--- Advanced Accuracy Report (WAPE %) ---")
print("Baseline (Seasonal Naive): ~24.75%")

lgbm_wape = calculate_wape(results_df['y'], results_df['LGBMRegressor'])
print(f"LightGBM Error:            {lgbm_wape:.2f}%")