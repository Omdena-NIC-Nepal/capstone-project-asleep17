import pandas as pd

def load_data(file_path="climate_daily_cleaned.csv"):
    df = pd.read_csv(file_path)
    return df

def classify_climate(row):
    # Tropical: High temp (above 30°C) and substantial rainfall (over 100mm)
    if row['Temp_2m'] > 30 and row['Precip'] > 100:
        return 'Tropical'
    # Temperate: Moderate temp (15°C to 30°C) and moderate rainfall (50mm to 100mm)
    elif 15 < row['Temp_2m'] <= 30 and 50 < row['Precip'] <= 100:
        return 'Temperate'
    # Cold: Low temp (below 15°C) and low rainfall (under 50mm)
    elif row['Temp_2m'] <= 15 and row['Precip'] <= 50:
        return 'Cold'
    else:
        return 'Other'  # This can be used to classify regions that don't fall into the standard categories