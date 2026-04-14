import duckdb
import urllib.request
import os
import time

print("Setting up data for Causal Inference...")

# --- 1. Download February 2019 Data & Zone Lookup ---
FEB_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-02.parquet"
FEB_FILE = "yellow_tripdata_2019-02.parquet"
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"
ZONE_FILE = "taxi_zone_lookup.csv"

if not os.path.exists(FEB_FILE):
    print("Downloading Feb 2019 Parquet (Post-Price Hike)...")
    urllib.request.urlretrieve(FEB_URL, FEB_FILE)

if not os.path.exists(ZONE_FILE):
    print("Downloading Taxi Zone Lookup CSV...")
    urllib.request.urlretrieve(ZONE_URL, ZONE_FILE)

# --- 2. Build the Causal Dataset in DuckDB ---
con = duckdb.connect('urbanflow.db')
print("Executing SQL: Joining Jan & Feb data with Borough boundaries...")

start_time = time.time()

# We combine Jan and Feb, join it with the Zone map, and aggregate to daily levels.
sql_causal_pipeline = f"""
CREATE OR REPLACE TABLE daily_borough_demand AS
WITH CombinedData AS (
    SELECT tpep_pickup_datetime, PULocationID, fare_amount
    FROM 'yellow_tripdata_2019-01.parquet'
    WHERE tpep_pickup_datetime >= '2019-01-01' AND tpep_pickup_datetime < '2019-02-01'
    UNION ALL
    SELECT tpep_pickup_datetime, PULocationID, fare_amount
    FROM '{FEB_FILE}'
    WHERE tpep_pickup_datetime >= '2019-02-01' AND tpep_pickup_datetime < '2019-03-01'
)
SELECT 
    date_trunc('day', c.tpep_pickup_datetime) AS ride_date,
    z.Borough AS borough,
    COUNT(*) AS total_trips,
    ROUND(AVG(c.fare_amount), 2) AS avg_fare
FROM CombinedData c
JOIN read_csv_auto('{ZONE_FILE}') z ON c.PULocationID = z.LocationID
WHERE 
    z.Borough IN ('Manhattan', 'Brooklyn')
    AND c.fare_amount > 0
GROUP BY 1, 2
ORDER BY 1, 2;
"""

con.execute(sql_causal_pipeline)
con.close()

execution_time = time.time() - start_time
print(f"Causal Data Pipeline finished in {execution_time:.2f} seconds.")
print("Database is ready for Econometric Modeling.")