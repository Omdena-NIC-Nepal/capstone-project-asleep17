Climate Data Analysis and Prediction Tool
This project aims to analyze, visualize, and predict climate change impacts using climate-related data. The application provides tools for exploratory data analysis (EDA), feature engineering, model predictions, and sentiment analysis of climate news articles.

Features
EDA Dashboard: Explore and visualize the climate data trends over decades, correlation heatmaps, and statistical tests for temperature changes across different periods.

Feature Engineering: Process climate data by creating new features, scaling the data, and applying Principal Component Analysis (PCA).

Model Prediction: Perform classification and regression tasks to predict climate zones and precipitation levels based on different climate factors.

Sentiment Analysis: Analyze the sentiment of climate-related news articles using natural language processing (NLP).

Requirements
Python 3.7+

Streamlit

Pandas

scikit-learn

Matplotlib

nltk

Other dependencies (can be installed via requirements.txt)

Setup
Clone this repository to your local machine:

bash
Copy
Edit
git clone https://github.com/yourusername/climate-data-analysis.git
Navigate to the project directory:

bash
Copy
Edit
cd climate-data-analysis
Install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the application:

bash
Copy
Edit
streamlit run app.py
Pages
EDA Dashboard
Data Preview: A preview of the climate data.

Temperature Trend: Visualize the average temperature and precipitation trends per decade.

Correlation Heatmap: View the correlation between different climate features like temperature, precipitation, humidity, and wind speed.

Statistical Test for Temperature Changes: Compare temperature changes between two periods using a T-test.

Feature Engineering
Feature Creation: Generate new features from the data, such as climate zones.

Data Scaling: Apply data scaling to standardize the dataset.

PCA: Perform Principal Component Analysis (PCA) for dimensionality reduction.

Model Prediction
Classification: Predict climate zones based on temperature, precipitation, and humidity.

Regression: Predict precipitation levels based on climate factors.

Sentiment Analysis
Sentiment Analysis of Climate News: Enter a climate-related news article, and the model will analyze the sentiment (positive, negative, or neutral).

File Structure
bash
Copy
Edit
.
├── app.py                # Main Streamlit app
├── utils/
|   |__ __init__.py
│   ├── data_loader.py    # Functions to load and preprocess the data
│   ├── feature_engineering.py # Functions for feature engineering
│   ├── modeling.py       # Functions for classification and regression models
│   ├── nlp.py            # Functions for sentiment analysis
│   └── visualization.py  # Functions for data visualization
|── app.py   # Python depe
|__ cleaned_glacier_data.csv
|__ climate_daily_cleaned.csv
|__ dailyclimate.csv
