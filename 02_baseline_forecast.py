import duckdb
import pandas as pd
import time
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoARIMA
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

print("Loading data from DuckDB...")
con = duckdb.connect('urbanflow.db')

# --- 1. Data Extraction & Formatting ---
# Nixtla libraries require a very specific DataFrame structure:
# unique_id (the series identifier), ds (the timestamp), and y (the target value)
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

# --- 2. Train/Test Split ---
# We have 31 days of data (Jan 2019). 
# We will train on the first 24 days and test our forecast on the last 7 days (168 hours).
horizon = 168 

# Sort to ensure chronological order, then split
df = df.sort_values(['unique_id', 'ds'])
train_df = df.groupby('unique_id').apply(lambda x: x.iloc[:-horizon]).reset_index(drop=True)
test_df = df.groupby('unique_id').apply(lambda x: x.iloc[-horizon:]).reset_index(drop=True)

print(f"Training on {len(train_df)} rows. Testing on {len(test_df)} rows.")

# --- 3. Define the Models ---
# Seasonality is 24 because our data is hourly and demand repeats daily
models = [
    SeasonalNaive(season_length=24),
    AutoARIMA(season_length=24, approximation=True) # Approximation speeds up compilation
]

# Instantiate the StatsForecast engine
sf = StatsForecast(
    models=models,
    freq='h', # Hourly frequency
    n_jobs=-1 # Use all available CPU cores
)

# --- 4. Train and Forecast ---
print("Training models and generating 7-day forecast. This may take a minute or two...")
start_time = time.time()

# Fit the models on the training data and predict the horizon length
forecast_df = sf.forecast(df=train_df, h=horizon)

execution_time = time.time() - start_time
print(f"Forecasting completed in {execution_time:.2f} seconds.")

# --- 5. Accuracy Tracking (The "Executive Defense") ---
# Merge predictions with actual test data to see how we did
results_df = forecast_df.reset_index().merge(test_df, on=['unique_id', 'ds'], how='left')

# Calculate WAPE (Weighted Average Percentage Error)
# WAPE is highly preferred in retail/marketplaces over standard MAPE because it handles zeroes better
def calculate_wape(y_true, y_pred):
    return (abs(y_true - y_pred).sum() / y_true.sum()) * 100

print("\n--- Accuracy Report (WAPE %) ---")
snaive_wape = calculate_wape(results_df['y'], results_df['SeasonalNaive'])
arima_wape = calculate_wape(results_df['y'], results_df['AutoARIMA'])

print(f"Seasonal Naive Error: {snaive_wape:.2f}%")
print(f"AutoARIMA Error:      {arima_wape:.2f}%")

# Save results for the dashboard later
results_df.to_csv('forecast_results.csv', index=False)
print("\nForecasts saved to 'forecast_results.csv'")