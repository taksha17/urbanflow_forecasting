import duckdb
import urllib.request
import os
import time

# --- 1. Download the Raw Data ---
# We are grabbing Jan 2019. This file is about 110MB and contains ~7.6 million rows.
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-01.parquet"
RAW_FILE = "yellow_tripdata_2019-01.parquet"

if not os.path.exists(RAW_FILE):
    print("Downloading raw Parquet file (this may take a minute)...")
    urllib.request.urlretrieve(DATA_URL, RAW_FILE)
    print("Download complete.")
else:
    print("Raw file already exists. Skipping download.")

# --- 2. Initialize DuckDB ---
# This creates a persistent local database file named 'urbanflow.db'
print("\nConnecting to DuckDB...")
con = duckdb.connect('urbanflow.db')

# --- 3. The SQL Pipeline ---
# We aggregate the 7.6 million individual rides into hourly buckets per Taxi Zone.
# Notice the WHERE clause: TLC data often has broken meters with dates in 2088 or 2001. 
# We filter strictly for Jan 2019 to ensure data quality.

print("Executing SQL Aggregation Pipeline...")
start_time = time.time()

sql_pipeline = f"""
-- Create or replace our clean, aggregated table
CREATE OR REPLACE TABLE hourly_zone_demand AS
SELECT 
    date_trunc('hour', tpep_pickup_datetime) AS pickup_hour,
    PULocationID AS pickup_zone_id,
    COUNT(*) AS total_trips,
    ROUND(AVG(fare_amount), 2) AS avg_fare,
    ROUND(AVG(trip_distance), 2) AS avg_distance
FROM '{RAW_FILE}'
WHERE 
    tpep_pickup_datetime >= '2019-01-01 00:00:00' 
    AND tpep_pickup_datetime < '2019-02-01 00:00:00'
    AND fare_amount > 0 -- Remove negative fares (refunds/errors)
    AND trip_distance > 0 -- Remove zero-distance trips
GROUP BY 
    1, 2
ORDER BY 
    1, 2;
"""

con.execute(sql_pipeline)
execution_time = time.time() - start_time
print(f"Pipeline finished in {execution_time:.2f} seconds.")

# --- 4. Verify the Results ---
print("\n--- Quick Sanity Check: Top 5 Rows ---")
preview = con.execute("SELECT * FROM hourly_zone_demand LIMIT 5").df()
print(preview)

print("\n--- Total Unique Time-Series Created ---")
# This tells us how many unique (Hour + Zone) combinations we have to forecast
count = con.execute("SELECT COUNT(*) FROM hourly_zone_demand").fetchone()[0]
print(f"Total rows in aggregated table: {count}")

con.close()