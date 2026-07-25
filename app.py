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
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #eef1f5 100%);
    }

    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 42px;
        font-weight: 700;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 16px;
        margin-bottom: 30px;
    }

    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #d1d5db;
        font-size: 15px;
        padding: 12px;
    }

    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 15px;
        transition: 0.3s;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
    }

    .result-card {
        padding: 20px;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 600;
        text-align: center;
        margin-top: 20px;
    }

    .real-card {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #34d399;
    }

    .fake-card {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
    }

    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ℹ️ About This Project")
    st.write("This tool uses **Natural Language Processing** and **Machine Learning** (TF-IDF + Logistic Regression) to classify news content.")
    st.metric("Model Accuracy", "98.94%")
    st.markdown("---")
    st.caption("Built as a final year project")

st.markdown('<p class="main-title">📰 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered NLP system to detect misinformation in news articles</p>', unsafe_allow_html=True)

user_input = st.text_area("", placeholder="Paste a news article or headline here...", height=200)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    check = st.button("🔍 Analyze News", use_container_width=True)

if check:
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        label, confidence = predict_news(user_input)
        if label == "REAL":
            st.markdown(f'<div class="result-card real-card">✅ This looks REAL<br><span style="font-size:14px;font-weight:400;">Confidence: {confidence}%</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-card fake-card">🚨 This looks FAKE<br><span style="font-size:14px;font-weight:400;">Confidence: {confidence}%</span></div>', unsafe_allow_html=True)
