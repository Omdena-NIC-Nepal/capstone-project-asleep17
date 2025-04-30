import pandas as pd
import gdown
import requests
import zipfile

# Function to download file from Google Drive using the file ID
def download_file_from_google_drive(file_id, output="climate_daily_cleaned.csv"):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)

# Function to load data from a given file path
def load_data(file_path="climate_daily_cleaned.csv"):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found locally. Downloading from Google Drive...")
        # File ID from Google Drive link
        file_id = '1kn4n6fezdE_JzuQxdXmfAdcApeTOMKR0'  # Replace with your file's ID
        download_file_from_google_drive(file_id)
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

df = load_data("climate_daily_cleaned.csv")

if df is not None:
    # Classify all climate types in the DataFrame
    df['Climate_Type'] = df.apply(classify_climate, axis=1)
    print(df.head())
