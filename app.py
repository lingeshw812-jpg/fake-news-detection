import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime

# ----------------------------
# NLP Setup
# ----------------------------
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


# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="The News Verifier", page_icon="🛡️", layout="centered")

# ----------------------------
# Session State
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "text_input_value" not in st.session_state:
    st.session_state.text_input_value = ""

SAMPLE_REAL = ("WASHINGTON (Reuters) - The Federal Reserve announced on Wednesday that it "
               "would raise interest rates by a quarter point, citing continued strength "
               "in the labor market and persistent inflation pressures. The decision was "
               "widely expected by economists.")

SAMPLE_FAKE = ("You won't BELIEVE what this celebrity said about the government! Shocking "
               "video reveals secret conspiracy that mainstream media doesn't want you to "
               "see. Share before it gets deleted!")


def fill_real():
    st.session_state.text_input_value = SAMPLE_REAL


def fill_fake():
    st.session_state.text_input_value = SAMPLE_FAKE


def clear_input():
    st.session_state.text_input_value = ""


# ----------------------------
# Global Styling (Navy / Purple Aurora Theme)
# ----------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Manrope:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #24243e, #302b63, #1a1a3d, #23213a);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399, #a78bfa);
        background-size: 300% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 4px;
        animation: shine 6s linear infinite;
        letter-spacing: 0.5px;
    }

    @keyframes shine { to { background-position: 300% center; } }

    .subtitle {
        text-align: center;
        color: #b8b8d1;
        font-size: 13px;
        margin-bottom: 10px;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .icon-row {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin: 25px 0 35px 0;
        flex-wrap: wrap;
    }

    .icon-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        font-size: 11px;
        color: #a5a5c0;
        letter-spacing: 0.5px;
        background: rgba(255,255,255,0.04);
        padding: 12px 18px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        min-width: 90px;
    }

    .icon-item span.emoji { font-size: 22px; margin-bottom: 6px; }

    /* ---- TEXTAREA FIX (high specificity + !important) ---- */
    .stApp textarea,
    .stApp div[data-testid="stTextArea"] textarea,
    .stApp div[data-baseweb="textarea"] textarea {
        background-color: #14142b !important;
        border-radius: 14px !important;
        border: 1px solid rgba(167, 139, 250, 0.4) !important;
        color: #f1f1f7 !important;
        -webkit-text-fill-color: #f1f1f7 !important;
        font-size: 15px !important;
        padding: 18px !important;
        caret-color: #a78bfa !important;
        opacity: 1 !important;
    }

    .stApp textarea::placeholder {
        color: #8888a8 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #8888a8 !important;
    }

    .stApp textarea:focus {
        border: 1px solid #a78bfa !important;
        box-shadow: 0 0 25px rgba(167, 139, 250, 0.3) !important;
    }

    .stButton>button {
        background: linear-gradient(90deg, #7c3aed, #2563eb, #7c3aed);
        background-size: 200% auto;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 13px 30px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.35s ease;
        letter-spacing: 0.8px;
    }

    .stButton>button:hover {
        background-position: right center;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.45);
    }

    .stButton>button:active { transform: translateY(0px) scale(0.97); }

    .ghost-btn button {
        background: transparent !important;
        border: 1px solid rgba(167,139,250,0.4) !important;
        color: #c4b5fd !important;
        box-shadow: none !important;
        font-size: 12px !important;
        padding: 8px 16px !important;
    }

    .signal-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 40px;
        animation: fadeIn 0.6s ease;
    }

    @keyframes fadeIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }

    .orb { width: 90px; height: 90px; border-radius: 50%; margin-bottom: 20px; }

    .orb-real {
        background: radial-gradient(circle at 35% 30%, #6ee7b7, #10b981 60%, #065f46);
        box-shadow: 0 0 20px #34d399, 0 0 50px rgba(52,211,153,0.5), 0 0 90px rgba(52,211,153,0.25);
        animation: pulseGreen 1.8s infinite ease-in-out;
    }

    .orb-fake {
        background: radial-gradient(circle at 35% 30%, #fca5a5, #ef4444 60%, #7f1d1d);
        box-shadow: 0 0 20px #f87171, 0 0 50px rgba(248,113,113,0.5), 0 0 90px rgba(248,113,113,0.25);
        animation: pulseRed 1.8s infinite ease-in-out;
    }

    @keyframes pulseGreen { 0%,100% { box-shadow: 0 0 20px #34d399, 0 0 50px rgba(52,211,153,0.5), 0 0 90px rgba(52,211,153,0.25); } 50% { box-shadow: 0 0 30px #34d399, 0 0 70px rgba(52,211,153,0.7), 0 0 120px rgba(52,211,153,0.4); } }
    @keyframes pulseRed { 0%,100% { box-shadow: 0 0 20px #f87171, 0 0 50px rgba(248,113,113,0.5), 0 0 90px rgba(248,113,113,0.25); } 50% { box-shadow: 0 0 30px #f87171, 0 0 70px rgba(248,113,113,0.7), 0 0 120px rgba(248,113,113,0.4); } }

    .signal-label { font-family: 'Playfair Display', serif; font-size: 28px; font-weight: 700; letter-spacing: 1px; }
    .label-real { color: #6ee7b7; }
    .label-fake { color: #fca5a5; }

    .badge {
        margin-top: 10px; font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase;
        color: #a5a5c0; border: 1px solid rgba(255,255,255,0.15); padding: 5px 14px; border-radius: 20px;
    }

    .conf-wrap { width: 260px; margin-top: 18px; }
    .conf-track { width: 100%; height: 6px; background: rgba(255,255,255,0.08); border-radius: 10px; overflow: hidden; }
    .conf-fill-real { height: 100%; background: linear-gradient(90deg, #10b981, #6ee7b7); border-radius: 10px; animation: fillBar 1.2s ease forwards; }
    .conf-fill-fake { height: 100%; background: linear-gradient(90deg, #ef4444, #fca5a5); border-radius: 10px; animation: fillBar 1.2s ease forwards; }
    @keyframes fillBar { from { width: 0%; } }
    .conf-text { text-align: center; margin-top: 10px; font-size: 13px; color: #a5a5c0; letter-spacing: 0.5px; }

    .section-title {
        font-family: 'Playfair Display', serif;
        color: #d8d8f0;
        font-size: 20px;
        font-weight: 700;
        margin-top: 55px;
        margin-bottom: 15px;
        text-align: center;
    }

    .history-item {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 13px;
        color: #cfcfe6;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .history-tag-real { color: #6ee7b7; font-weight: 700; }
    .history-tag-fake { color: #fca5a5; font-weight: 700; }

    .stats-row {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-top: 20px;
        flex-wrap: wrap;
    }

    .stat-box { text-align: center; }
    .stat-num {
        font-family: 'Playfair Display', serif;
        font-size: 28px;
        font-weight: 800;
        color: #a78bfa;
    }
    .stat-label {
        font-size: 11px;
        color: #a5a5c0;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    .footer-note {
        text-align: center;
        color: #6b6b8a;
        font-size: 12px;
        margin-top: 60px;
        margin-bottom: 10px;
        letter-spacing: 0.5px;
    }

    section[data-testid="stSidebar"] {
        background: #0d0d1f;
        border-right: 1px solid rgba(167,139,250,0.25);
    }
    section[data-testid="stSidebar"] * { color: #d8d8f0 !important; }
    div[data-testid="stMetricValue"] { color: #a78bfa !important; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("### 🛡️ About This Project")
    st.write(
        "This tool uses **Natural Language Processing** and **Machine Learning** "
        "(TF-IDF + Logistic Regression) to classify news content as real or fake."
    )
    st.metric("Model Accuracy", "98.94%")
    st.markdown("---")
    st.markdown("### ⚙️ How It Works")
    st.write("1. Text is cleaned and lemmatized")
    st.write("2. Converted to TF-IDF vectors")
    st.write("3. Classified by trained model")
    st.write("4. Confidence score calculated")
    st.markdown("---")
    st.caption("Built as a final year project")

# ----------------------------
# Header
# ----------------------------
st.markdown('<p class="main-title">🛡️ The News Verifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Detection of Misinformation</p>', unsafe_allow_html=True)

st.markdown("""
    <div class="icon-row">
        <div class="icon-item"><span class="emoji">🔍</span>NLP Analysis</div>
        <div class="icon-item"><span class="emoji">🧠</span>ML Powered</div>
        <div class="icon-item"><span class="emoji">⚡</span>Instant Result</div>
        <div class="icon-item"><span class="emoji">🔒</span>Secure Check</div>
    </div>
""", unsafe_allow_html=True)

# ----------------------------
# Input Area
# ----------------------------
user_input = st.text_area(
    "",
    value=st.session_state.text_input_value,
    placeholder="Paste a news article or headline here...",
    height=200,
    key="main_textarea"
)

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col2:
    st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
    st.button("Try Real Example", on_click=fill_real, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
    st.button("Try Fake Example", on_click=fill_fake, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
    st.button("Clear", on_click=clear_input, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
colA, colB, colC = st.columns([1, 1, 1])
with colB:
    check = st.button("Verify News", use_container_width=True)

# ----------------------------
# Prediction + Result Display
# ----------------------------
if check:
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing content..."):
            label, confidence = predict_news(user_input)

        st.session_state.history.insert(0, {
            "text": user_input[:60] + ("..." if len(user_input) > 60 else ""),
            "label": label,
            "confidence": confidence,
            "time": datetime.now().strftime("%H:%M:%S")
        })
        st.session_state.history = st.session_state.history[:5]

        if label == "REAL":
            st.markdown(f"""
                <div class="signal-wrap">
                    <div class="orb orb-real"></div>
                    <div class="signal-label label-real">VERIFIED REAL</div>
                    <div class="badge">AI Confidence Score</div>
                    <div class="conf-wrap">
                        <div class="conf-track"><div class="conf-fill-real" style="width:{confidence}%;"></div></div>
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
                        <div class="conf-track"><div class="conf-fill-fake" style="width:{confidence}%;"></div></div>
                        <div class="conf-text">{confidence}% Confidence</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# ----------------------------
# Recent Checks History
# ----------------------------
if st.session_state.history:
    st.markdown('<p class="section-title">Recent Checks</p>', unsafe_allow_html=True)
    for item in st.session_state.history:
        tag_class = "history-tag-real" if item["label"] == "REAL" else "history-tag-fake"
        st.markdown(f"""
            <div class="history-item">
                <span>{item['text']}</span>
                <span class="{tag_class}">{item['label']} · {item['confidence']}% · {item['time']}</span>
            </div>
        """, unsafe_allow_html=True)

# ----------------------------
# Stats Footer
# ----------------------------
st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)
st.markdown("""
    <div class="stats-row">
        <div class="stat-box"><div class="stat-num">98.94%</div><div class="stat-label">Accuracy</div></div>
        <div class="stat-box"><div class="stat-num">44,898</div><div class="stat-label">Articles Trained</div></div>
        <div class="stat-box"><div class="stat-num">5,000</div><div class="stat-label">TF-IDF Features</div></div>
        <div class="stat-box"><div class="stat-num">2</div><div class="stat-label">ML Models Compared</div></div>
    </div>
""", unsafe_allow_html=True)

st.markdown('<p class="footer-note">Built with NLP + Machine Learning · Final Year Project</p>', unsafe_allow_html=True)
import requests

NEWS_API_KEY = "6a900bc79d5d4b7caf3490ca397426cf"  # paste your key from newsapi.org

def fetch_live_headlines(country="us", category="general", count=5):
    url = f"https://newsapi.org/v2/top-headlines?country={country}&category={category}&apiKey={NEWS_API_KEY}&pageSize={count}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        articles = data.get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", "") or "",
                "source": a.get("source", {}).get("name", "Unknown")
            }
            for a in articles
        ]
    return []
    st.markdown('<p class="section-title">Check Live Headlines</p>', unsafe_allow_html=True)

if st.button("📡 Fetch Latest Real News", use_container_width=True):
    with st.spinner("Fetching live headlines..."):
        headlines = fetch_live_headlines()

    if headlines:
        for h in headlines:
            combined_text = h["title"] + " " + h["description"]
            label, confidence = predict_news(combined_text)
            tag_class = "history-tag-real" if label == "REAL" else "history-tag-fake"
            st.markdown(f"""
                <div class="history-item">
                    <span>{h['title']} <br><small style="color:#888;">Source: {h['source']}</small></span>
                    <span class="{tag_class}">{label} · {confidence}%</span>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Could not fetch live news. Check your API key or try again.")
        
