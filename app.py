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

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Manrope:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a3d);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .main-title {
        font-family: 'Sora', sans-serif;
        font-size: 46px;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        animation: shine 4s linear infinite;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 16px;
        margin-bottom: 35px;
        font-weight: 400;
        opacity: 0.85;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        animation: fadeInUp 0.8s ease;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #f1f5f9 !important;
        font-size: 15px !important;
        padding: 14px !important;
    }

    .stTextArea textarea::placeholder {
        color: #94a3b8 !important;
    }

    .stButton>button {
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 36px;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
        letter-spacing: 0.3px;
    }

    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 30px rgba(124, 58, 237, 0.6);
    }

    .stButton>button:active {
        transform: translateY(0px) scale(0.98);
    }

    .result-card {
        padding: 28px;
        border-radius: 18px;
        font-size: 22px;
        font-weight: 700;
        text-align: center;
        margin-top: 25px;
        animation: popIn 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        backdrop-filter: blur(10px);
    }

    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.8); }
        100% { opacity: 1; transform: scale(1); }
    }

    .real-card {
        background: rgba(52, 211, 153, 0.12);
        color: #6ee7b7;
        border: 1px solid rgba(52, 211, 153, 0.4);
        box-shadow: 0 0 30px rgba(52, 211, 153, 0.15);
    }

    .fake-card {
        background: rgba(248, 113, 113, 0.12);
        color: #fca5a5;
        border: 1px solid rgba(248, 113, 113, 0.4);
        box-shadow: 0 0 30px rgba(248, 113, 113, 0.15);
    }

    .confidence-text {
        font-size: 14px;
        font-weight: 400;
        opacity: 0.75;
        margin-top: 6px;
    }

    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        backdrop-filter: blur(10px);
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
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

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

user_input = st.text_area("", placeholder="Paste a news article or headline here...", height=200)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    check = st.button("🔍 Analyze News", use_container_width=True)

if check:
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing content..."):
            label, confidence = predict_news(user_input)
        if label == "REAL":
            st.markdown(f'<div class="result-card real-card">✅ This looks REAL<div class="confidence-text">Confidence: {confidence}%</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-card fake-card">🚨 This looks FAKE<div class="confidence-text">Confidence: {confidence}%</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
