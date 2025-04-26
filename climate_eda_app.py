import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, classification_report


# Load Data
def load_data():
    df = pd.read_csv("climate_daily_cleaned.csv")
    return df  

df = load_data()

# Adding the 'Climate_Zone' classification logic here
def classify_climate(row):
    if row['Temp_2m'] > 30 and row['Precip'] > 100:
        return 'Tropical'
    elif row['Temp_2m'] > 15 and row['Precip'] > 50:
        return 'Temperate'
    else:
        return 'Cold'

# Apply the function to create the 'Climate_Zone' column
df['Climate_Zone'] = df.apply(classify_climate, axis=1)



# Rest of your analysis follows...
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
    st.subheader("1. Derived Climate Indices")
    # Calculate Drought Index (Avoid division by NaN or zero values)
    df['Rolling_Mean'] = df['Precip'].rolling(window=30).mean()
    df['Rolling_Std'] = df['Precip'].rolling(window=30).std()

    # Drought Index: (Precip - Rolling Mean) / Rolling Std
    df['Drought_Index'] = (df['Precip'] - df['Rolling_Mean']) / df['Rolling_Std']

# Fill NaN values that might appear due to rolling window
    df['Drought_Index'] = df['Drought_Index'].fillna(0)  # You can choose to fill with 0 or use forward fill

    st.write(df[['Date', 'Precip', 'Drought_Index']].head())

    df['Is_Monsoon'] = df['Date'].dt.month.isin([6, 7, 8, 9]).astype(int)
    st.write(df[['Date', 'Is_Monsoon']].head())

    st.subheader("3. Lag Features")
    df['Temp_Lag1'] = df['Temp_2m'].shift(1)
    df['Precip_Lag7'] = df['Precip'].shift(7)
    st.write(df[['Temp_Lag1', 'Precip_Lag7']].head())

    st.subheader("4. Scaling Features")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['Temp_2m', 'Precip', 'Humidity_2m', 'Pressure', 'WindSpeed_10m']])
    scaled_df = pd.DataFrame(scaled_features, columns=['Temp_2m_scaled', 'Precip_scaled', 'Humidity_2m_scaled', 'Pressure_scaled', 'WindSpeed_10m_scaled'])
    df = pd.concat([df, scaled_df], axis=1)
    st.write(scaled_df.head())

    st.subheader("5. Dimensionality Reduction")
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(principal_components, columns=['PCA1', 'PCA2'])
    df = pd.concat([df, pca_df], axis=1)
    st.write(pca_df.head())


elif page == "Model Prediction":
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Temp_2m_scaled'] = StandardScaler().fit_transform(df[['Temp_2m']])

# 1. Classification Example (Random Forest for Climate Zone Classification) 
    X_class = df[['Temp_2m', 'Precip', 'Humidity_2m']]  # Example feature selection
    y_class = df['Climate_Zone']  # Replace with actual climate zone classification label
    
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

    rf_clf = RandomForestClassifier(n_estimators=100)
    rf_clf.fit(X_train_class, y_train_class)
    y_pred_class = rf_clf.predict(X_test_class)

# Evaluate the classification model
    st.write("Random Forest Classification Model Evaluation")
    st.write(f"F1 Score: {f1_score(y_test_class, y_pred_class, average='weighted')}")
    st.write(f"Classification Report:\n{classification_report(y_test_class, y_pred_class)}")

    X_reg = df[['Temp_2m_scaled', 'Precip', 'Humidity_2m']]  # Example feature selection
    y_reg = df['Precip']  # Target variable: Precipitation

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    gbr_reg = GradientBoostingRegressor(n_estimators=100)
    gbr_reg.fit(X_train_reg, y_train_reg)
    y_pred_reg = gbr_reg.predict(X_test_reg)
    
    st.write("Gradient Boosting Regression Model Evaluation")
    st.write(f"RMSE: {mean_squared_error(y_test_reg, y_pred_reg, squared=False)}")
    st.write(f"MAE: {mean_absolute_error(y_test_reg, y_pred_reg)}")
    st.write(f"R²: {gbr_reg.score(X_test_reg, y_test_reg)}")

# Evaluate the regression model
    cross_val_score_rf = cross_val_score(rf_clf, X_class, y_class, cv=5)
    cross_val_score_gbr = cross_val_score(gbr_reg, X_reg, y_reg, cv=5)

    st.write("Cross-validation Scores for Random Forest:", cross_val_score_rf)
    st.write("Cross-validation Scores for Gradient Boosting Regression:", cross_val_score_gbr)

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

