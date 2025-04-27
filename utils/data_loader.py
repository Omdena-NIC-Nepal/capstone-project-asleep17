import pandas as pd

def load_data(file_path="climate_daily_cleaned.csv"):
    df = pd.read_csv(file_path)
    return df

def classify_climate(row):
    if row['Temp_2m'] > 30 and row['Precip'] > 100:
        return 'Tropical'
    elif row['Temp_2m'] > 15 and row['Precip'] > 50:
        return 'Temperate'
    else:
        return 'Cold'
