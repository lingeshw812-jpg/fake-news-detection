import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load saved model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def predict_news(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    label = "REAL" if prediction == 1 else "FAKE"
    confidence = round(max(probability) * 100, 2)
    return label, confidence

# --- Streamlit UI ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection using NLP")
st.write("Paste a news article or headline below to check if it's Real or Fake.")

user_input = st.text_area("Enter news text here:", height=200)

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        label, confidence = predict_news(user_input)
        if label == "REAL":
            st.success(f"✅ Prediction: {label} (Confidence: {confidence}%)")
        else:
            st.error(f"🚨 Prediction: {label} (Confidence: {confidence}%)")
