├── app.py # Main Streamlit app for the dashboard and analysis ├── utils/ │ ├── data_loader.py # Functions for loading and processing data │ ├── feature_engineering.py # Feature engineering methods │ ├── modeling.py # Modeling (Classification and Regression) │ ├── visualization.py # Functions for plotting graphs and charts │ └── nlp.py # Functions for natural language processing (sentiment analysis, topic modeling) ├── requirements.txt # Python dependencies for the project ├── data/ # Folder for storing raw and processed data │ └── climate_data.csv # Sample climate data file (replace with your own) └── README.md # Project documentation (this file)

markdown
Copy
Edit
### web app link:https://capstone-project-asleep17-ts3o5st9ds3wzpgwkmqbi6.streamlit.app/
## Features

1. **EDA Dashboard**: 
   - Visualize the trends of temperature and precipitation over time.
   - Generate a correlation heatmap of various climate variables.
   - Conduct statistical tests to determine significant temperature changes between different time periods.

2. **Feature Engineering**: 
   - Create new features like drought index, monsoon period, lag features, etc.
   - Scale features and apply Principal Component Analysis (PCA) to reduce dimensionality.

3. **Model Prediction**: 
   - **Classification**: Classify the climate zone based on temperature and precipitation.
   - **Regression**: Predict next day's precipitation using regression models.

4. **Sentiment Analysis**: 
   - Analyze sentiment in climate-related news articles.

5. **Topic Modeling**: 
   - Identify key topics in climate-related text data using topic modeling techniques.

6. **Text Summarization**: 
   - Generate summaries of long climate reports.

## Installation

To run the project locally, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/climate-change-analysis.git
cd climate-change-analysis
2. Set up a virtual environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Run the Streamlit app
bash
Copy
Edit
streamlit run app.py
The app will open in your default web browser.

Usage
EDA Dashboard: Select the "EDA Dashboard" from the sidebar to view the visualizations.

Feature Engineering: Select "Feature Engineering" to see the data processing steps.

Model Prediction: Select "Model Prediction" to view classification and regression model results.

Sentiment Analysis: Select "Sentiment Analysis" to analyze the sentiment of climate-related news articles.

Topic Modeling: Select "Topic Modeling" to perform topic modeling on climate articles.

Text Summarization: Select "Text Summarization" to summarize climate reports.

Technologies Used
Python: Programming language used for the analysis and modeling.

Streamlit: Web framework for building the interactive app.

Pandas, NumPy: Libraries for data manipulation and analysis.

Scikit-learn: Machine learning library for model training and evaluation.

Matplotlib, Seaborn: Libraries for data visualization.

NLTK, TextBlob: Libraries for natural language processing tasks.