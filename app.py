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
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Manrope:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: radial-gradient(circle at top, #1a1a1a 0%, #000000 70%);
    }

    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 50px;
        font-weight: 800;
        background: linear-gradient(90deg, #d4af37, #f4e5c2, #d4af37);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 4px;
        animation: shine 5s linear infinite;
        letter-spacing: 0.5px;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    .subtitle {
        text-align: center;
        color: #a8a8a8;
        font-size: 15px;
        margin-bottom: 40px;
        font-weight: 400;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    div[data-testid="stTextArea"] textarea {
        background-color: #141414 !important;
        border-radius: 14px !important;
        border: 1px solid #d4af37 !important;
        color: #f5f5f5 !important;
        font-size: 15px !important;
        padding: 16px !important;
        caret-color: #d4af37 !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: #777777 !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border: 1px solid #f4e5c2 !important;
        box-shadow: 0 0 15px rgba(212, 175, 55, 0.3) !important;
    }

    .stButton>button {
        background: linear-gradient(90deg, #b8860b, #d4af37, #b8860b);
        background-size: 200% auto;
        color: #0a0a0a;
        border: none;
        border-radius: 12px;
        padding: 14px 36px;
        font-weight: 700;
        font-size: 15px;
        transition: all 0.4s ease;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .stButton>button:hover {
        background-position: right center;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(212, 175, 55, 0.4);
    }

    .stButton>button:active {
        transform: translateY(0px) scale(0.98);
    }

    .result-card {
        padding: 28px;
        border-radius: 16px;
        font-size: 22px;
        font-weight: 700;
        text-align: center;
        margin-top: 25px;
        animation: popIn 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    }

    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.85); }
        100% { opacity: 1; transform: scale(1); }
    }

    .real-card {
        background: rgba(52, 211, 153, 0.08);
        color: #6ee7b7;
        border: 1px solid rgba(52, 211, 153, 0.4);
        box-shadow: 0 0 25px rgba(52, 211, 153, 0.12);
    }

    .fake-card {
        background: rgba(212, 175, 55, 0.08);
        color: #e8c766;
        border: 1px solid rgba(212, 175, 55, 0.4);
        box-shadow: 0 0 25px rgba(212, 175, 55, 0.12);
    }

    .confidence-text {
        font-size: 14px;
        font-weight: 400;
        opacity: 0.75;
        margin-top: 6px;
    }

    section[data-testid="stSidebar"] {
        background: #0a0a0a;
        border-right: 1px solid #d4af37;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e5e5 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #d4af37 !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ℹ️ About This Project")
    st.write("This tool uses **Natural Language Processing** and **Machine Learning** (TF-IDF + Logistic Regression) to classify news content.")
    st.metric("Model Accuracy", "98.94%")
    st.markdown("---")
    st.caption("Built as a final year project")

st.markdown('<p class="main-title">📰 The News Verifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Detection of Misinformation</p>', unsafe_allow_html=True)

user_input = st.text_area("", placeholder="Paste a news article or headline here...", height=200)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    check = st.button("Analyze News", use_container_width=True)

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
