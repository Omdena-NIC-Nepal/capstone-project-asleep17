import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import ttest_ind

# Load Data
def load_data():
    df = pd.read_csv("climate_daily_cleaned.csv")
    return df  

df = load_data()
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Decade'] = (df['Year'] // 10) * 10
decade_avg = df.groupby('Decade')[['Temp_2m', 'Precip']].mean().reset_index()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", [
    "EDA Dashboard",
    "Feature Engineering",
    "Model Prediction",
    "Download Reports",
    "User Feedback"
])

# ================== Pages ======================

if page == "EDA Dashboard":
    st.title("Climate Data EDA Dashboard")
    
    st.header("1. Data Preview")
    st.write(df.head())

    st.header("2. Temperature Trend Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=decade_avg, x='Decade', y='Temp_2m', ax=ax)
    ax.set_title("Average Temperature per Decade")
    st.pyplot(fig)

    st.header("3. Precipitation Trend Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=decade_avg, x='Decade', y='Precip', ax=ax)
    ax.set_title("Average Precipitation per Decade")
    st.pyplot(fig)

    st.header("4. Extreme Weather Events")
    extreme_temp = df[df['Temp_2m'] > 35]
    extreme_precip = df[df['Precip'] > 100]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=extreme_temp, x='Decade', ax=ax, palette='coolwarm')
    ax.set_title("Frequency of Extreme Temperature Events")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=extreme_precip, x='Decade', ax=ax, palette='Blues')
    ax.set_title("Frequency of Extreme Precipitation Events")
    st.pyplot(fig)

    st.header("5. Correlation Between Climate Variables")
    corr_matrix = df[['Temp_2m', 'Precip', 'Humidity_2m', 'Pressure', 'WindSpeed_10m']].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    st.header("6. Statistical Test for Temperature Changes")
    period1_start = st.date_input("Start Date for Period 1", value=pd.to_datetime("1980-01-01"))
    period1_end = st.date_input("End Date for Period 1", value=pd.to_datetime("1999-12-31"))
    period2_start = st.date_input("Start Date for Period 2", value=pd.to_datetime("2000-01-01"))
    period2_end = st.date_input("End Date for Period 2", value=pd.to_datetime("2020-12-31"))

    period1_data = df[(df['Date'] >= str(period1_start)) & (df['Date'] <= str(period1_end))]['Temp_2m']
    period2_data = df[(df['Date'] >= str(period2_start)) & (df['Date'] <= str(period2_end))]['Temp_2m']

    t_stat, p_val = ttest_ind(period1_data, period2_data)

    st.write(f"T-statistic: {t_stat}")
    st.write(f"P-value: {p_val}")

    if p_val < 0.05:
        st.success("There is a significant change in temperature between the two periods.")
    else:
        st.info("No significant change in temperature between the two periods.")

# ------------- OTHER PAGES ----------------

elif page == "Feature Engineering":
    st.title("Feature Engineering Page")
    st.write("🚧 Coming soon: Derived indices, lag features, PCA, etc.")

elif page == "Model Prediction":
    st.title("🔮 Climate Forecast Models Coming Soon!")
    st.write("🚧 Work in progress... Stay tuned.")

elif page == "Download Reports":
    st.title("📄 Download Analysis Report")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Full Climate Data (CSV)",
        data=csv,
        file_name='climate_data_processed.csv',
        mime='text/csv',
    )

elif page == "User Feedback":
    st.title("💬 User Feedback")
    feedback = st.text_area("Please provide your feedback here:")
    if st.button("Submit"):
        st.success("Thank you for your feedback!")

