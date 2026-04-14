import duckdb
import pandas as pd
import torch
import time
import numpy as np
from chronos import ChronosPipeline

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

# Impute missing hours (same as LightGBM script)
all_zones = df['unique_id'].unique()
all_hours = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='h')
full_idx = pd.MultiIndex.from_product([all_zones, all_hours], names=['unique_id', 'ds'])
df = df.set_index(['unique_id', 'ds']).reindex(full_idx, fill_value=0).reset_index()

# Train/Test Split
horizon = 168 
df = df.sort_values(['unique_id', 'ds'])
test_df = df.groupby('unique_id').tail(horizon)
train_df = df.drop(test_df.index)

# --- Step 2. Initialize the Foundation Model ---
print("\nDownloading and loading Amazon Chronos-T5-Mini from Hugging Face...")
# We use device_map="cpu" to run it locally without needing a massive GPU
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-mini",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

# --- 3. Generate Zero-Shot Forecasts ---
print(f"Generating Zero-Shot AI forecast for {len(all_zones)} zones. This may take a moment on CPU...")
start_time = time.time()

chronos_forecasts = []

# Chronos processes series individually or in batches. We loop through our zones.
for zone in all_zones:
    zone_data = train_df[train_df['unique_id'] == zone]['y'].values
    
    # Convert context to a PyTorch tensor
    context_tensor = torch.tensor(zone_data)
    
    # Generate 168 hours into the future
    # Chronos outputs multiple sample paths (probabilistic); we take the median path
    forecast_samples = pipeline.predict(context_tensor, horizon)
    median_forecast = np.quantile(forecast_samples[0].numpy(), 0.5, axis=0)
    
    # Store results
    zone_test_ds = test_df[test_df['unique_id'] == zone]['ds']
    zone_results = pd.DataFrame({
        'unique_id': zone,
        'ds': zone_test_ds,
        'Chronos': median_forecast
    })
    chronos_forecasts.append(zone_results)

chronos_df = pd.concat(chronos_forecasts)

execution_time = time.time() - start_time
print(f"Foundation Model Forecasting completed in {execution_time:.2f} seconds.")

# --- 4. Combine All Forecasts & Track Accuracy ---
print("\nMerging Chronos forecasts with previous LightGBM results...")

# Load the previous results that contain the LightGBM forecasts
previous_results = pd.read_csv('forecast_results.csv')
previous_results['ds'] = pd.to_datetime(previous_results['ds'])

# THE FIX: Force unique_id to be a string in both dataframes before merging
previous_results['unique_id'] = previous_results['unique_id'].astype(str)
chronos_df['unique_id'] = chronos_df['unique_id'].astype(str)
chronos_df['ds'] = pd.to_datetime(chronos_df['ds'])

# Merge the new Chronos forecasts onto the existing ones
results_df = previous_results.merge(chronos_df, on=['unique_id', 'ds'], how='left')

def calculate_wape(y_true, y_pred):
    if y_true.sum() == 0: return 0
    return (abs(y_true - y_pred).sum() / y_true.sum()) * 100

chronos_wape = calculate_wape(results_df['y'], results_df['Chronos'])

print("\n--- Final AI Architecture Report (WAPE %) ---")
print("Baseline (Seasonal Naive): ~24.75%")
print("Machine Learning (LightGBM): ~18.09%")
print(f"Foundation Model (Chronos-T5): {chronos_wape:.2f}%")

# Save the final 3-way comparison!
results_df.to_csv('final_forecasts.csv', index=False)
print("Saved complete 3-way comparison to 'final_forecasts.csv'")