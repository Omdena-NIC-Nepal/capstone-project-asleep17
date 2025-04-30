import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data
from utils.feature_engineering import add_climate_zone, create_features, scale_features, apply_pca
from utils.modeling import classification_model, regression_model
from utils.visualization import plot_trend, plot_extreme_events, plot_correlation_heatmap,perform_t_test
from utils.nlp import analyze_sentiment

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
    "Sentiment Analysis",
    "Topic Modeling",
    "Text Summarization",
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

    # Prepare data
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['Temp_2m', 'Precip', 'Humidity_2m', 'Pressure', 'WindSpeed_10m'])

    # 1. Create features
    df_fe = create_features(df.copy())
    st.subheader("Data After Feature Creation")
    st.write(df_fe[['Temp_2m', 'Precip', 'Drought_Index', 'Is_Monsoon', 'Temp_Lag1', 'Precip_Lag7']].head())

    # 2. Select features to scale
    feature_cols = ['Temp_2m', 'Precip', 'Humidity_2m', 'Pressure', 'WindSpeed_10m',
                    'Drought_Index', 'Temp_Lag1', 'Precip_Lag7']

    df_clean = df_fe[feature_cols].dropna()
    st.subheader("Selected Features for Scaling")
    st.write("Shape after dropping NaNs:", df_clean.shape)
    st.write(df_clean.head())

    if df_clean.empty:
        st.warning("⚠️ Feature engineered DataFrame is empty after dropping NaNs.")
    else:
        # 3. Scale features
        scaled = scale_features(df_clean)
        scaled_df = pd.DataFrame(scaled, columns=feature_cols)
        st.subheader("Scaled Features")
        st.write(scaled_df.head())

        # 4. PCA
        pca_result = apply_pca(scaled)
        pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
        st.subheader("PCA Result")
        st.write(pca_df.head())

        st.success("✅ Feature Engineering Complete")



elif page == "Model Prediction":
    st.title("Modeling")

    # Classification
    X_class = df[['Temp_2m', 'Precip', 'Humidity_2m']]
    y_class = df['Climate_Zone']

    clf, f1, report = classification_model(X_class, y_class)
    st.subheader("Classification Results")
    st.write(f"F1 Score: {f1}")
    st.text(report)

    # Regression: Predicting next day's precipitation
    df['Precip_tomorrow'] = df['Precip'].shift(-1)
    df = df.dropna()
    X_reg = df[['Temp_2m', 'Precip', 'Humidity_2m']]
    y_reg = df['Precip_tomorrow']

    reg, rmse, mae, r2 = regression_model(X_reg, y_reg)
    st.subheader("Regression Results")
    st.write(f"RMSE: {rmse}")
    st.write(f"MAE: {mae}")
    st.write(f"R2 Score: {r2}")

elif page == "Topic Modeling":
    st.header("Topic Modeling for Climate Articles")
    
    # User input: Upload text file or input text
    uploaded_file = st.file_uploader("Upload Climate Articles (Text file)", type=["txt"])
    text_input = st.text_area("Or enter your own climate article here:")

    if uploaded_file is not None:
        # Read the uploaded text file
        text_data = uploaded_file.read().decode("utf-8").splitlines()
    elif text_input:
        text_data = [text_input]
    else:
        st.warning("Please upload a file or enter a text.")
        text_data = []

    if text_data:
        # Perform topic modeling
        topics = topic_modeling(text_data)
        st.write("Identified Topics:")
        for topic in topics:
            st.write(topic)

elif page == "Text Summarization":
    st.header("Summarize Climate Report")
    
    # User input: Upload climate report or enter text
    uploaded_report = st.file_uploader("Upload Climate Report (Text file)", type=["txt"])
    report_input = st.text_area("Or enter your own climate report here:")

    if uploaded_report is not None:
        # Read the uploaded report
        report_text = uploaded_report.read().decode("utf-8")
    elif report_input:
        report_text = report_input
    else:
        st.warning("Please upload a report or enter your text.")
        report_text = ""

    if report_text:
        # Generate summary
        summary = summarize_text(report_text)
        st.subheader("Summary:")
        st.write(summary)

elif page == "Sentiment Analysis":
    st.title("Sentiment Analysis of Climate News")

    # Input for news article
    text_input = st.text_area("Enter climate-related news article here:")

    if text_input:
        # Analyze the sentiment
        sentiment_score, sentiment = analyze_sentiment(text_input)

        # Display the result
        if sentiment == 'positive':
            st.success(f"The sentiment is positive with a polarity score of {sentiment_score:.2f}")
        elif sentiment == 'negative':
            st.error(f"The sentiment is negative with a polarity score of {sentiment_score:.2f}")
        else:
            st.info(f"The sentiment is neutral with a polarity score of {sentiment_score:.2f}")