import string
import nltk
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Download NLTK stopwords if not already installed
nltk.download('stopwords')

# Preprocessing function for text
def preprocess_text(text):
    """
    Preprocess text by lowering the case, removing punctuation, and filtering stop words.
    """
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Function to perform Topic Modeling using LDA
def topic_modeling(text_data):
    """
    Perform topic modeling using Latent Dirichlet Allocation (LDA).
    Returns the top topics in the form of words.
    """
    processed_data = [preprocess_text(text) for text in text_data]
    
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(processed_data)
    
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X)
    
    def print_top_words(model, feature_names, n_top_words):
        """
        Extract the top words for each topic.
        """
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            topics.append(f"Topic {topic_idx + 1}: " + " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        return topics

    return print_top_words(lda, vectorizer.get_feature_names_out(), 5)

# Function to summarize a text (climate report) by extracting the top sentences
def summarize_text(report_text):
    """
    Summarizes the input text by extracting the top-ranked sentences.
    """
    sentences = sent_tokenize(report_text)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    
    sentence_scores = np.array(X.sum(axis=1)).flatten()
    ranked_sentences = [sentences[i] for i in sentence_scores.argsort()[-3:][::-1]]  # Top 3 sentences
    
    summary = ' '.join(ranked_sentences)
    return summary

# Function to analyze sentiment using TextBlob
def analyze_sentiment(text):
    """
    Analyzes the sentiment of the provided text.
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
