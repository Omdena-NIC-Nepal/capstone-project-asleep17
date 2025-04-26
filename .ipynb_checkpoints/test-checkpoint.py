import pandas as pd

# Load Data
def load_data():
    df = pd.read_csv("climate_daily_cleaned.csv")
    return df  

df = load_data()

# Check column names to see if 'Climate_Zone' exists
print(df.columns)  # This will print the list of all columns in your dataframe

