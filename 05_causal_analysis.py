import duckdb
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading daily demand data from DuckDB...")
con = duckdb.connect('urbanflow.db')

# --- Step 1. Load Data ---
df = con.execute("SELECT * FROM daily_borough_demand").df()
con.close()

# --- Step 2. Feature Engineering for Difference-in-Differences ---
# The price hike happened on Feb 2, 2019.
# Treatment Group: Manhattan (subject to the surcharge)
# Control Group: Brooklyn (exempt from the surcharge)

df['ride_date'] = pd.to_datetime(df['ride_date'])

# Create our dummy variables for the regression
df['is_post_hike'] = (df['ride_date'] >= '2019-02-02').astype(int)
df['is_manhattan'] = (df['borough'] == 'Manhattan').astype(int)

# We use the natural log of total trips. 
# This is a classic economics trick so our coefficients represent percentage changes (elasticity).
df['log_trips'] = np.log(df['total_trips'])

# --- Step 3. The Econometric Model ---
# This formula runs the DiD regression.
# The interaction term (is_manhattan:is_post_hike) isolates the true Causal Impact.
print("\nRunning Causal Inference Model (Difference-in-Differences)...")
model = smf.ols('log_trips ~ is_manhattan + is_post_hike + is_manhattan:is_post_hike', data=df).fit()

# Print the executive summary
print("\n--- Executive Econometrics Summary ---")
print(model.summary().tables[1])

# Extract the causal impact coefficient
causal_impact = model.params['is_manhattan:is_post_hike']
percent_change = (np.exp(causal_impact) - 1) * 100

print(f"\n[CAUSAL ESTIMATE]: The $2.50 congestion surcharge caused a {percent_change:.2f}% shift in Manhattan taxi demand relative to the Brooklyn baseline.")

# ---Step 4. The Executive Visual (Defending the Forecast) ---
print("\nGenerating Executive Dashboard Visual...")
plt.figure(figsize=(12, 6))

# Plot the raw trends
sns.lineplot(data=df, x='ride_date', y='total_trips', hue='borough', marker='o')

# Add the intervention line
plt.axvline(pd.to_datetime('2019-02-02'), color='red', linestyle='--', label='Price Hike ($2.50 Surcharge)')

plt.title('Causal Impact: NYC Taxi Congestion Surcharge ($2.50)', fontsize=14, pad=15)
plt.xlabel('Date', fontweight='bold')
plt.ylabel('Daily Total Trips', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('causal_impact_dashboard.png', dpi=300)
print("Saved visualization as 'causal_impact_dashboard.png'.")