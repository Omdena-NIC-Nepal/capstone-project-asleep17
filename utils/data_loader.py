import pandas as pd
import pandas as pd
import gdown
import os

# Function to download file from Google Drive using the file ID
def download_file_from_google_drive(file_id, output="climate_daily_cleaned.csv"):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)

# Function to load data from a given file path
def load_data(file_path="climate_daily_cleaned.csv"):
    print(f"Checking if file {file_path} exists...")
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found locally. Downloading from Google Drive...")
        file_id = '1kn4n6fezdE_JzuQxdXmfAdcApeTOMKR0'  # Replace with your file's ID
        download_file_from_google_drive(file_id, file_path)  # Download the file
        print(f"File downloaded: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# Sample usage



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
df = load_data()

if df is not None:
    print(df.head())
else:
    print("Failed to load data.")


