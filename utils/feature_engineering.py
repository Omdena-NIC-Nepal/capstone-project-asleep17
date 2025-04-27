import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def add_climate_zone(df):
    from utils.data_loader import classify_climate
    df['Climate_Zone'] = df.apply(classify_climate, axis=1)
    return df

def create_features(df):
    df['Rolling_Mean'] = df['Precip'].rolling(window=30).mean()
    df['Rolling_Std'] = df['Precip'].rolling(window=30).std()
    df['Drought_Index'] = (df['Precip'] - df['Rolling_Mean']) / df['Rolling_Std']
    df['Drought_Index'] = df['Drought_Index'].fillna(0)
    df['Is_Monsoon'] = df['Date'].dt.month.isin([6,7,8,9]).astype(int)
    df['Temp_Lag1'] = df['Temp_2m'].shift(1)
    df['Precip_Lag7'] = df['Precip'].shift(7)
    return df

def scale_features(df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['Temp_2m', 'Precip', 'Humidity_2m', 'Pressure', 'WindSpeed_10m']])
    return scaled_features

def apply_pca(scaled_features):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    return principal_components
