import streamlit as st
import pandas as pd
from PIL import Image

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="UrbanFlow | AI Forecasting & Causality",
    page_icon="🚕",
    layout="wide"
)

st.title("🚕 UrbanFlow: AI & Causal Inference Engine")
st.markdown("""
Welcome to the UrbanFlow executive dashboard. This project demonstrates high-volume hierarchical time-series forecasting, benchmarking traditional ML against state-of-the-art LLMs, alongside econometric causal inference.
""")

# --- 2. Load Data ---
@st.cache_data
def load_forecast_data():
    try:
        # Loading the new 3-way data file
        df = pd.read_csv('final_forecasts.csv')
        df['ds'] = pd.to_datetime(df['ds'])
        return df
    except FileNotFoundError:
        return None

df_forecast = load_forecast_data()

# --- 3. Dashboard Tabs ---
tab1, tab2 = st.tabs(["🤖 AI vs ML Engine", "⚖️ Causal Pricing Inference"])

# --- TAB 1: Forecasting ---
with tab1:
    st.header("Model Benchmarking: Compute vs Accuracy")
    st.markdown("Comparing gradient-boosted trees (LightGBM) against zero-shot foundation models (Amazon Chronos-T5).")
    
    # Executive Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Baseline (Seasonal Naive)", value="24.75% Error")
    col2.metric(label="LightGBM (Feature Engineered)", value="18.09% Error", delta="-27% vs Base", delta_color="inverse")
    col3.metric(label="Chronos-T5 (Zero-Shot AI)", value="11.10% Error", delta="-55% vs Base", delta_color="inverse")
    
    st.markdown("---")
    st.markdown("### Zone-Level Forecast vs Actuals")
    
    if df_forecast is not None:
        # Let the user select a specific taxi zone
        zones = df_forecast['unique_id'].unique()
        selected_zone = st.selectbox("Select Taxi Zone ID to View:", zones, index=0)
        
        # Filter data for the chart
        zone_data = df_forecast[df_forecast['unique_id'] == selected_zone]
        
        # Plot a 3-way line chart
        chart_data = zone_data.set_index('ds')[['y', 'LGBMRegressor', 'Chronos']]
        chart_data.columns = ['Actual Demand', 'LightGBM Forecast', 'Chronos-T5 Forecast']
        
        st.line_chart(
            data=chart_data, 
            color=["#FF4B4B", "#0068C9", "#00C94B"] # Red, Blue, Green
        )
        st.caption("Red: Actual | Blue: LightGBM (1-sec compute) | Green: Chronos (300-sec compute)")
    else:
        st.warning("⚠️ `final_forecasts.csv` not found. Please run the foundation model script and push the CSV to GitHub!")

# --- TAB 2: Causal Inference ---
with tab2:
    st.header("Pricing Elasticity: The $2.50 Surcharge Impact")
    st.markdown("""
    **Methodology:** Difference-in-Differences (DiD)  
    **Treatment:** Manhattan (Subject to $2.50 Congestion Surcharge on Feb 2, 2019)  
    **Control:** Brooklyn (Exempt from Surcharge)
    """)
    
    st.info("💡 **Executive Summary:** The econometric regression isolated the demand shock of the price hike, proving a statistically significant drop in Manhattan trip volume relative to the baseline.")
    
    try:
        img = Image.open('causal_impact_dashboard.png')
        st.image(img, use_container_width=True)
    except FileNotFoundError:
        st.warning("⚠️ `causal_impact_dashboard.png` not found. Upload it to GitHub!")