import streamlit as st
import pandas as pd
import time

from backend.model_loader import load_model
from backend.predictor import predict_air_quality
from backend.esp_fetch import fetch_esp32_data
from backend.charts import *
from backend.metrics import display_metrics

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Air Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AI-Based Smart Air Quality Monitoring System")
st.caption("ESP32 | Machine Learning | Real-Time Analysis")

# ==========================================================
# SESSION STATES
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history=[]

if "esp32_connected" not in st.session_state:
    st.session_state.esp32_connected=False

if "sensor_data" not in st.session_state:
    st.session_state.sensor_data=None

# ==========================================================
# LOAD MODEL
# ==========================================================

pipeline,scaler,iso_forest,feature_cols=load_model()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Control Panel")

page=st.sidebar.radio(
    "Navigation",
    [
        "Live Dashboard",
        "Manual Prediction",
        "Analytics",
        "History"
    ]
)

refresh_rate=st.sidebar.slider(
    "Refresh Interval (s)",
    1,
    10,
    3
)

# ==========================================================
# STATUS BAR
# ==========================================================

c1,c2,c3=st.columns(3)

with c1:
    if st.session_state.esp32_connected:
        st.success("ESP32 Connected")
    else:
        st.error("ESP32 Offline")

with c2:
    st.info("ML Model Loaded")

with c3:
    st.write(
        "Last Update:",
        time.strftime("%H:%M:%S")
    )
# ==========================================================
# LIVE DASHBOARD
# ==========================================================

if page=="Live Dashboard":

    ip=st.text_input(
        "ESP32 IP Address",
        "192.168.1.5"
    )

    auto_refresh=st.checkbox("Auto Refresh")

    if st.button("Fetch Data") or auto_refresh:

        data=fetch_esp32_data(ip)

        if data:

            st.session_state.sensor_data=data
            st.session_state.esp32_connected=True

        elif st.session_state.sensor_data:

            st.warning("Using previous data")

        else:

            st.session_state.esp32_connected=False
            st.error("ESP32 Offline")

        if st.session_state.sensor_data:

            temp=st.session_state.sensor_data["temp"]
            hum=st.session_state.sensor_data["hum"]
            air=st.session_state.sensor_data["air"]
            dust=st.session_state.sensor_data["dust"]

            score,category,color,anomaly=\
            predict_air_quality(
                temp,
                hum,
                air,
                dust,
                scaler,
                iso_forest,
                feature_cols
            )

            c1,c2,c3,c4=st.columns(4)

            c1.metric(
                "Temperature",
                f"{temp:.1f} °C"
            )

            c2.metric(
                "Humidity",
                f"{hum:.1f} %"
            )

            c3.metric(
                "Air Quality",
                f"{air:.1f} ppm"
            )

            c4.metric(
                "Dust",
                f"{dust:.1f} µg/m³"
            )

            st.plotly_chart(
                gauge_chart(score,color),
                width="stretch"
            )

            if anomaly==-1:
                st.warning("Abnormal Pattern Detected")

            st.success(category)

            st.session_state.history.append({

                "Time":time.strftime("%H:%M:%S"),
                "Temperature":temp,
                "Humidity":hum,
                "AirQuality":air,
                "Dust":dust,
                "AI_Score":score,
                "Category":category

            })

        if auto_refresh:

            time.sleep(refresh_rate)
            st.rerun()


# ==========================================================
# MANUAL PREDICTION
# ==========================================================

if page=="Manual Prediction":

    c1,c2=st.columns(2)

    with c1:

        temp=st.slider(
            "Temperature (°C)",
            15.0,
            45.0,
            28.0
        )

        hum=st.slider(
            "Humidity (%)",
            20,
            100,
            70
        )

    with c2:

        air=st.number_input(
            "Air Quality (ppm)",
            0,
            5000,
            300
        )

        dust=st.number_input(
            "Dust Density (µg/m³)",
            0,
            500,
            100
        )

    source=st.selectbox(
        "Environment",
        [
            "indoor",
            "outdoor",
            "kitchen"
        ]
    )

    if st.button("Predict"):

        score,category,color,anomaly=\
        predict_air_quality(
            temp,
            hum,
            air,
            dust,
            scaler,
            iso_forest,
            feature_cols,
            source
        )

        st.plotly_chart(
            gauge_chart(score,color),
            width="stretch"
        )

        if anomaly==-1:
            st.warning("Abnormal Pattern Detected")

        st.success(category)
# ==========================================================
# ANALYTICS
# ==========================================================

if page=="Analytics":

    st.subheader("Model Performance")

    display_metrics(st)

    st.subheader("Feature Importance")

    st.plotly_chart(
        feature_importance_chart(),
        width="stretch"
    )

    st.subheader("Air Quality Distribution")

    st.plotly_chart(
        category_distribution_chart(),
        width="stretch"
    )


# ==========================================================
# HISTORY
# ==========================================================

if page=="History":

    if len(st.session_state.history)>0:

        df=pd.DataFrame(
            st.session_state.history
        )

        st.subheader("History")

        st.dataframe(
            df,
            width="stretch"
        )

        st.subheader("AI Score Trend")

        st.plotly_chart(
            trend_chart(df),
            width="stretch"
        )

        csv=df.to_csv(index=False)

        st.download_button(
            "Download CSV",
            csv,
            file_name="history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):

            st.session_state.history=[]

            st.rerun()

    else:

        st.info("No History Available")


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "AI-Based Smart Air Quality Monitoring System | "
    "ESP32 • Machine Learning • Streamlit"
)
