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

st.set_page_config(page_title="The News Verifier", page_icon="📰", layout="centered")

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
        background: #0a0a0c;
        background-image:
            radial-gradient(circle at 20% 20%, rgba(212, 175, 55, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 80% 80%, rgba(52, 211, 153, 0.06) 0%, transparent 40%),
            radial-gradient(circle at 50% 50%, rgba(10, 10, 12, 1) 0%, #000000 100%);
        background-attachment: fixed;
    }

    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 48px;
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
        color: #8a8a8a;
        font-size: 13px;
        margin-bottom: 45px;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    div[data-testid="stTextArea"] textarea {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(212, 175, 55, 0.35) !important;
        color: #f5f5f5 !important;
        font-size: 15px !important;
        padding: 18px !important;
        caret-color: #d4af37 !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: #666666 !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border: 1px solid #d4af37 !important;
        box-shadow: 0 0 25px rgba(212, 175, 55, 0.25) !important;
    }

    .stButton>button {
        background: linear-gradient(90deg, #b8860b, #d4af37, #b8860b);
        background-size: 200% auto;
        color: #0a0a0a;
        border: none;
        border-radius: 12px;
        padding: 14px 40px;
        font-weight: 700;
        font-size: 14px;
        transition: all 0.4s ease;
        letter-spacing: 1.5px;
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

    /* Signal Orb */
    .signal-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 40px;
        animation: fadeIn 0.6s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .orb {
        width: 90px;
        height: 90px;
        border-radius: 50%;
        margin-bottom: 20px;
        position: relative;
    }

    .orb-real {
        background: radial-gradient(circle at 35% 30%, #6ee7b7, #10b981 60%, #065f46);
        box-shadow: 0 0 20px #34d399, 0 0 50px rgba(52, 211, 153, 0.5), 0 0 90px rgba(52, 211, 153, 0.25);
        animation: pulseGreen 1.8s infinite ease-in-out;
    }

    .orb-fake {
        background: radial-gradient(circle at 35% 30%, #fca5a5, #ef4444 60%, #7f1d1d);
        box-shadow: 0 0 20px #f87171, 0 0 50px rgba(248, 113, 113, 0.5), 0 0 90px rgba(248, 113, 113, 0.25);
        animation: pulseRed 1.8s infinite ease-in-out;
    }

    @keyframes pulseGreen {
        0%, 100% { box-shadow: 0 0 20px #34d399, 0 0 50px rgba(52, 211, 153, 0.5), 0 0 90px rgba(52, 211, 153, 0.25); }
        50% { box-shadow: 0 0 30px #34d399, 0 0 70px rgba(52, 211, 153, 0.7), 0 0 120px rgba(52, 211, 153, 0.4); }
    }

    @keyframes pulseRed {
        0%, 100% { box-shadow: 0 0 20px #f87171, 0 0 50px rgba(248, 113, 113, 0.5), 0 0 90px rgba(248, 113, 113, 0.25); }
        50% { box-shadow: 0 0 30px #f87171, 0 0 70px rgba(248, 113, 113, 0.7), 0 0 120px rgba(248, 113, 113, 0.4); }
    }

    .signal-label {
        font-family: 'Playfair Display', serif;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .label-real { color: #6ee7b7; }
    .label-fake { color: #fca5a5; }

    .badge {
        margin-top: 10px;
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #666;
        border: 1px solid #333;
        padding: 5px 14px;
        border-radius: 20px;
    }

    /* Confidence bar */
    .conf-wrap {
        width: 260px;
        margin-top: 18px;
    }

    .conf-track {
        width: 100%;
        height: 6px;
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
        overflow: hidden;
    }

    .conf-fill-real {
        height: 100%;
        background: linear-gradient(90deg, #10b981, #6ee7b7);
        border-radius: 10px;
        animation: fillBar 1.2s ease forwards;
    }

    .conf-fill-fake {
        height: 100%;
        background: linear-gradient(90deg, #ef4444, #fca5a5);
        border-radius: 10px;
        animation: fillBar 1.2s ease forwards;
    }

    @keyframes fillBar {
        from { width: 0%; }
    }

    .conf-text {
        text-align: center;
        margin-top: 10px;
        font-size: 13px;
        color: #999;
        letter-spacing: 0.5px;
    }

    section[data-testid="stSidebar"] {
        background: #050505;
        border-right: 1px solid rgba(212, 175, 55, 0.3);
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
    check = st.button("Verify News", use_container_width=True)

if check:
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing content..."):
            label, confidence = predict_news(user_input)

        if label == "REAL":
            st.markdown(f"""
                <div class="signal-wrap">
                    <div class="orb orb-real"></div>
                    <div class="signal-label label-real">VERIFIED REAL</div>
                    <div class="badge">AI Confidence Score</div>
                    <div class="conf-wrap">
                        <div class="conf-track">
                            <div class="conf-fill-real" style="width:{confidence}%;"></div>
                        </div>
                        <div class="conf-text">{confidence}% Confidence</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="signal-wrap">
                    <div class="orb orb-fake"></div>
                    <div class="signal-label label-fake">FLAGGED FAKE</div>
                    <div class="badge">AI Confidence Score</div>
                    <div class="conf-wrap">
                        <div class="conf-track">
                            <div class="conf-fill-fake" style="width:{confidence}%;"></div>
                        </div>
                        <div class="conf-text">{confidence}% Confidence</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
