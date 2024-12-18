import streamlit as st
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib

# Load the trained ensemble model
ensemble_model = joblib.load('/Users/akhjan/Documents/KBTU Courses/Machine Learning/sentiment_analysis_model.pkl')

# Load the pre-trained embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'\<br.*?\>', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Define aspect mapping
aspect_mapping = {
    'User Satisfaction': 'User satisfaction and happiness',
    'Product Experience': 'Product usability and features',
    'General Experience': 'Overall experience and opinions'
}

# Generate aspect-weighted embeddings
def generate_aspect_weighted_embeddings(texts, aspects):
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    aspect_embeddings = embedding_model.encode(aspects, show_progress_bar=False)
    weighted_embeddings = []
    for text_emb, aspect_emb in zip(embeddings, aspect_embeddings):
        combined_embedding = np.concatenate([text_emb, aspect_emb])  # Concatenate embeddings
        weighted_embeddings.append(combined_embedding)
    return np.array(weighted_embeddings)

# Predict sentiment
def predict_sentiment(user_input, aspect_category):
    cleaned_text = clean_text(user_input)
    aspect_text = aspect_mapping.get(aspect_category, "General experience and opinions")
    embeddings = generate_aspect_weighted_embeddings([cleaned_text], [aspect_text])
    prediction = ensemble_model.predict(embeddings)
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_map[prediction[0]]

# Streamlit UI
st.title("Sentiment Analysis with Aspect Weighting")

# User Input
user_input = st.text_area("Enter your review text:", "")
aspect_category = st.selectbox(
    "Select an aspect category:",
    ["User Satisfaction and Happiness", "Product Experience (usability and features)", "Overall experience and opinions"]
)

# Analyze Button
if st.button("Analyze Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input, aspect_category)

        # Change text color based on sentiment
        if sentiment == 'Positive':
            st.markdown(f"<h3 style='color: green;'>Predicted Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
        elif sentiment == 'Negative':
            st.markdown(f"<h3 style='color: red;'>Predicted Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
        else:  # Neutral
            st.markdown(f"<h3 style='color: blue;'>Predicted Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
    else:
        st.error("Please enter a review text to analyze.")
