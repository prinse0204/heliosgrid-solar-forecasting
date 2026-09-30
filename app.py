import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import shap
import streamlit as st

# Configure the Streamlit page layout and title
st.set_page_config(
    layout="wide",
    page_title="HeliosGrid | Solar Power Intelligence",
)

# Custom dark theme styling for the dashboard
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1E293B;
    }

    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #FFB800 0%, #F59E0B 50%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    div[data-testid="stMetric"] {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B;
        border-radius: 8px;
        color: #94A3B8;
        padding: 8px 16px;
        border: 1px solid #334155;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFB800 !important;
        color: #0F172A !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("HELIOSGRID: Solar Forecasting & Grid Telemetry")
st.caption("Industrial Machine Learning Pipeline - XGBoost Predictor - Live Weather API Sync")

# Load trained model artifacts from the models directory
MODEL_PATH = os.path.join("models", "solar_xgboost_artifacts.pkl")


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


artifacts = load_artifacts()

if artifacts is None:
    st.error(
        "Model file not found. Please run the training pipeline first to generate 'models/solar_xgboost_artifacts.pkl'."
    )
    st.stop()

model = artifacts["model"]
last_val = artifacts["last_observed_value"]
target_name = artifacts["target_col_name"]
feature_names = artifacts["features"]

# Geographic coordinates for target microgrid sites
CITIES_COORDS = {
    "Bhadla Solar Park, Rajasthan": (27.5386, 71.9161),
    "Mumbai, India": (19.0760, 72.8777),
    "Delhi NCR, India": (28.6139, 77.2090),
    "Bengaluru, India": (12.9716, 77.5946),
    "Jodhpur, India": (26.2389, 73.0243),
    "Berlin, Germany": (52.5200, 13.4050),
    "Madrid, Spain": (40.4168, -3.7038),
}


# Fetch current weather conditions from Open-Meteo
def get_live_solar_irradiance(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=cloudcover"
        response = requests.get(url, timeout=5).json()
        cloud_cover = response["hourly"]["cloudcover"][0]
        irradiance_factor = max(0.1, (100 - cloud_cover) / 100.0)
        return irradiance_factor, cloud_cover, None
    except Exception as e:
        return 1.0, 0, str(e)


# Sidebar layout for user parameters and simulation triggers
st.sidebar.title("Control Panel")
grid_demand_base = st.sidebar.slider("Base Grid Demand (kW/MW)", 50, 1000, 250)

st.sidebar.markdown("---")
st.sidebar.subheader("Telemetry Sync")
auto_mode = st.sidebar.checkbox("Enable Live Weather Sync", value=True)

if auto_mode:
    selected_city = st.sidebar.selectbox("Microgrid Location", list(CITIES_COORDS.keys()))
    lat, lon = CITIES_COORDS[selected_city]
    irradiance_factor, cloud_cover, err = get_live_solar_irradiance(lat, lon)
    if err:
        st.sidebar.warning("Live Weather Sync Offline. Defaulting to 100%.")
    else:
        st.sidebar.info(f"Location: {selected_city}\n\nCloud Cover: {cloud_cover}%")
else:
    cloud_cover = st.sidebar.slider("Simulated Cloud Cover (%)", 0, 100, 0)
    irradiance_factor = max(0.1, (100 - cloud_cover) / 100.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Fault Diagnostics")
inject_anomaly = st.sidebar.checkbox("Simulate Inverter Fault (12 PM)", value=False)

# Build a 24-hour future forecast horizon using the trained model
now = pd.Timestamp.now().floor("h")
future_times = pd.date_range(start=now, periods=24, freq="h")

predictions = []
current_lag1 = last_val
feature_matrix = []

for dt in future_times:
    hour_sin = np.sin(2 * np.pi * dt.hour / 24.0)
    hour_cos = np.cos(2 * np.pi * dt.hour / 24.0)

    feature_vector = [
        hour_sin, hour_cos, dt.month, dt.dayofweek,
        current_lag1, current_lag1 * 0.95, current_lag1 * 0.90, current_lag1
    ]
    feature_matrix.append(feature_vector)
    pred = model.predict(np.array([feature_vector]))[0]

    # Zero out predictions outside daylight hours
    if 6 <= dt.hour <= 18:
        pred = max(0.0, pred) * irradiance_factor
    else:
        pred = 0.0

    predictions.append(pred)
    if pred > 0:
        current_lag1 = pred

X_forecast = pd.DataFrame(feature_matrix, columns=feature_names)

forecast_df = pd.DataFrame({
    "Time": future_times,
    "Forecasted Generation": predictions,
    "Simulated Demand": [
        grid_demand_base + np.sin(t.hour * np.pi / 12) * (grid_demand_base * 0.25)
        for t in future_times
    ]
})

# Apply dynamic fault injection if selected
if inject_anomaly:
    for i in range(len(forecast_df)):
        if forecast_df.loc[i, "Time"].hour == 12:
            forecast_df.loc[i, "Forecasted Generation"] *= 0.05

# Scan forecast output for uncharacteristic drops during peak hours
detected_anomalies = []
for i in range(1, len(forecast_df)):
    prev_val = forecast_df.loc[i - 1, "Forecasted Generation"]
    curr_val = forecast_df.loc[i, "Forecasted Generation"]
    time_slot = forecast_df.loc[i, "Time"]

    if 10 <= time_slot.hour <= 15 and prev_val > 10.0:
        drop_ratio = (prev_val - curr_val) / prev_val
        if drop_ratio > 0.60 and cloud_cover < 70:
            detected_anomalies.append((time_slot, drop_ratio * 100))

# Display summary KPI cards
col1, col2, col3 = st.columns(3)
col1.metric("24h Peak Output", f"{forecast_df['Forecasted Generation'].max():.1f} kW")
col2.metric("24h Total Generation", f"{forecast_df['Forecasted Generation'].sum():.1f} kWh")

is_stable = forecast_df["Forecasted Generation"].sum() >= forecast_df["Simulated Demand"].sum() * 0.4
col3.metric("Grid Capacity", "OPTIMAL" if is_stable else "DEFICIT")

if len(detected_anomalies) > 0:
    for time_slot, drop_pct in detected_anomalies:
        st.error(
            f"CRITICAL ANOMALY ALERT [{time_slot.strftime('%H:%M')}]: Sudden power drop of {drop_pct:.1f}% detected under clear skies. Inverter fault suspected."
        )

st.markdown("<br>", unsafe_allow_html=True)

# Main dashboard tab structure
tab1, tab2 = st.tabs(["Generation Telemetry", "Explainable AI (SHAP)"])

with tab1:
    fig = go.Figure()

    # Solar power generation curve
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Time"],
            y=forecast_df["Forecasted Generation"],
            name="Solar Power Output",
            line=dict(color="#FFB800", width=3.5),
            fill="tozeroy",
            fillcolor="rgba(255, 184, 0, 0.08)",
        )
    )

    # Grid load demand curve
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Time"],
            y=forecast_df["Simulated Demand"],
            name="Grid Load Curve",
            line=dict(color="#00F2FE", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#E2E8F0", family="Inter"),
        xaxis=dict(gridcolor="#1E293B", showgrid=True),
        yaxis=dict(
            gridcolor="#1E293B", showgrid=True, title=f"Power ({target_name})"
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        margin=dict(l=20, r=20, t=30, b=20),
        height=420,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Hourly predictions data table
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("View Hourly Prediction Data Table"):
        st.dataframe(forecast_df, use_container_width=True)

with tab2:
    st.markdown("Model Decision Attribution (SHAP Analysis)")
    st.caption("TreeSHAP analysis breaking down XGBoost feature weight distribution.")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_forecast)

        mean_shap = np.abs(shap_values.values).mean(axis=0)
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": mean_shap
        }).sort_values(by="Importance", ascending=True)

        fig_shap = go.Figure(go.Bar(
            x=shap_df["Importance"],
            y=shap_df["Feature"],
            orientation="h",
            marker=dict(color="#00F2FE")
        ))

        fig_shap.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            font=dict(color="#E2E8F0", family="Inter"),
            xaxis=dict(gridcolor="#1E293B", title="Mean |SHAP Contribution|"),
            yaxis=dict(gridcolor="#1E293B"),
            height=380,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    except Exception as e:
        st.warning(f"SHAP visualization error: {e}")