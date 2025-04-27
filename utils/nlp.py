# nlp.py
from textblob import TextBlob

def analyze_sentiment(text):
    """
    Analyze the sentiment of the provided text using TextBlob.
    Returns the polarity score and sentiment type (positive, negative, neutral).
    """
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    
    if sentiment_score > 0:
        sentiment = 'positive'
    elif sentiment_score < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return sentiment_score, sentiment
