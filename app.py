import streamlit as st
import pandas as pd

from utils.data_loader import load_data
from utils.feature_engineering import add_climate_zone, create_features, scale_features, apply_pca
from utils.modeling import classification_model, regression_model
from utils.visualization import plot_trend, plot_extreme_events, plot_correlation_heatmap,perform_t_test


# Load Data
df = load_data()
df['Date'] = pd.to_datetime(df['Date'])
df = add_climate_zone(df)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", [
    "EDA Dashboard",
    "Feature Engineering",
    "Model Prediction",
])

# Pages
if page == "EDA Dashboard":
    st.title("Climate Data EDA Dashboard")
    
    st.header("Data Preview")
    st.write(df.head())

    st.header("Temperature Trend")
    df['Year'] = df['Date'].dt.year
    df['Decade'] = (df['Year'] // 10) * 10
    decade_avg = df.groupby('Decade')[['Temp_2m', 'Precip']].mean().reset_index()
    
    fig = plot_trend(decade_avg, 'Decade', 'Temp_2m', "Average Temperature per Decade")
    st.pyplot(fig)

    fig = plot_trend(decade_avg, 'Decade', 'Precip', "Average Precipitation per Decade")
    st.pyplot(fig)

    st.header("Correlation Heatmap")
    fig = plot_correlation_heatmap(df, ['Temp_2m', 'Precip', 'Humidity_2m', 'Pressure', 'WindSpeed_10m'])
    st.pyplot(fig)
    
     # Statistical Test for Temperature Changes
    st.header("Statistical Test for Temperature Changes")
    period1_start = st.date_input("Start Date for Period 1", value=pd.to_datetime("1980-01-01"))
    period1_end = st.date_input("End Date for Period 1", value=pd.to_datetime("1999-12-31"))
    period2_start = st.date_input("Start Date for Period 2", value=pd.to_datetime("2000-01-01"))
    period2_end = st.date_input("End Date for Period 2", value=pd.to_datetime("2020-12-31"))

    # Call the function from visualization.py
    t_stat, p_val = perform_t_test(df, 'Date', 'Temp_2m', period1_start, period1_end, period2_start, period2_end)

    st.write(f"T-statistic: {t_stat}")
    st.write(f"P-value: {p_val}")

    if p_val < 0.05:
        st.success("There is a significant change in temperature between the two periods.")
    else:
        st.info("No significant change in temperature between the two periods.")

elif page == "Feature Engineering":
    st.title("Feature Engineering")
    
    df = create_features(df)
    scaled = scale_features(df)
    principal_components = apply_pca(scaled)

    st.write("Feature Engineering Done ✅")

elif page == "Model Prediction":
    st.title("Modeling")

    X_class = df[['Temp_2m', 'Precip', 'Humidity_2m']]
    y_class = df['Climate_Zone']

    clf, f1, report = classification_model(X_class, y_class)
    st.subheader("Classification Results")
    st.write(f"F1 Score: {f1}")
    st.text(report)

    X_reg = df[['Temp_2m', 'Precip', 'Humidity_2m']]
    y_reg = df['Precip']

    reg, rmse, mae, r2 = regression_model(X_reg, y_reg)
    st.subheader("Regression Results")
    st.write(f"RMSE: {rmse}")
    st.write(f"MAE: {mae}")
    st.write(f"R2 Score: {r2}")
