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

st.set_page_config(page_title="The News Verifier", page_icon="🛡️", layout="centered")

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

is_dark = st.session_state.theme == "dark"

if is_dark:
    bg_main = "#0a0a0c"
    bg_glow = "radial-gradient(circle at 20% 20%, rgba(212,175,55,0.08) 0%, transparent 40%), radial-gradient(circle at 80% 80%, rgba(52,211,153,0.06) 0%, transparent 40%), radial-gradient(circle at 50% 50%, #0a0a0c 0%, #000000 100%)"
    text_color = "#f5f5f5"
    input_bg = "rgba(255,255,255,0.04)"
    input_border = "rgba(212,175,55,0.35)"
    subtitle_color = "#8a8a8a"
    placeholder_color = "#777777"
    sidebar_bg = "#050505"
else:
    bg_main = "#faf8f3"
    bg_glow = "radial-gradient(circle at 20% 20%, rgba(212,175,55,0.10) 0%, transparent 40%), radial-gradient(circle at 80% 80%, rgba(16,185,129,0.08) 0%, transparent 40%), radial-gradient(circle at 50% 50%, #faf8f3 0%, #f0ece0 100%)"
    text_color = "#1a1a1a"
    input_bg = "#ffffff"
    input_border = "rgba(180,140,20,0.5)"
    subtitle_color = "#555555"
    placeholder_color = "#999999"
    sidebar_bg = "#f5f1e8"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Manrope:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Manrope', sans-serif;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .stApp {{
        background: {bg_main};
        background-image: {bg_glow};
        background-attachment: fixed;
    }}

    .main-title {{
        font-family: 'Playfair Display', serif;
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(90deg, #b8860b, #d4af37, #b8860b);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 4px;
        animation: shine 5s linear infinite;
        letter-spacing: 0.5px;
    }}

    @keyframes shine {{ to {{ background-position: 200% center; }} }}

    .subtitle {{
        text-align: center;
        color: {subtitle_color};
        font-size: 13px;
        margin-bottom: 10px;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
    }}

    .icon-row {{
        display: flex;
        justify-content: center;
        gap: 28px;
        margin-bottom: 35px;
        opacity: 0.75;
    }}

    .icon-item {{
        display: flex;
        flex-direction: column;
        align-items: center;
        font-size: 11px;
        color: {subtitle_color};
        letter-spacing: 0.5px;
    }}

    .icon-item span.emoji {{
        font-size: 22px;
        margin-bottom: 4px;
    }}

    /* Force textarea styling — override browser/streamlit defaults */
    textarea, .stTextArea textarea, div[data-testid="stTextArea"] textarea {{
        background-color: {input_bg} !important;
        border-radius: 14px !important;
        border: 1px solid {input_border} !important;
        color: {text_color} !important;
        font-size: 15px !important;
        padding: 18px !important;
        -webkit-text-fill-color: {text_color} !important;
        opacity: 1 !important;
    }}

    textarea::placeholder {{
        color: {placeholder_color} !important;
        opacity: 1 !important;
    }}

    textarea:focus {{
        border: 1px solid #d4af37 !important;
        box-shadow: 0 0 25px rgba(212, 175, 55, 0.25) !important;
    }}

    .stButton>button {{
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
    }}

    .stButton>button:hover {{
        background-position: right center;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(212, 175, 55, 0.4);
    }}

    .theme-toggle button {{
        background: transparent !important;
        color: {text_color} !important;
        border: 1px solid {input_border} !important;
        padding: 6px 14px !important;
        font-size: 12px !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        box-shadow: none !important;
    }}

    .signal-wrap {{
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 40px;
        animation: fadeIn 0.6s ease;
    }}

    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(15px); }} to {{ opacity: 1; transform: translateY(0); }} }}

    .orb {{
        width: 90px;
        height: 90px;
        border-radius: 50%;
        margin-bottom: 20px;
    }}

    .orb-real {{
        background: radial-gradient(circle at 35% 30%, #6ee7b7, #10b981 60%, #065f46);
        box-shadow: 0 0 20px #34d399, 0 0 50px rgba(52,211,153,0.5), 0 0 90px rgba(52,211,153,0.25);
        animation: pulseGreen 1.8s infinite ease-in-out;
    }}

    .orb-fake {{
        background: radial-gradient(circle at 35% 30%, #fca5a5, #ef4444 60%, #7f1d1d);
        box-shadow: 0 0 20px #f87171, 0 0 50px rgba(248,113,113,0.5), 0 0 90px rgba(248,113,113,0.25);
        animation: pulseRed 1.8s infinite ease-in-out;
    }}

    @keyframes pulseGreen {{ 0%,100% {{ box-shadow: 0 0 20px #34d399, 0 0 50px rgba(52,211,153,0.5), 0 0 90px rgba(52,211,153,0.25); }} 50% {{ box-shadow: 0 0 30px #34d399, 0 0 70px rgba(52,211,153,0.7), 0 0 120px rgba(52,211,153,0.4); }} }}
    @keyframes pulseRed {{ 0%,100% {{ box-shadow: 0 0 20px #f87171, 0 0 50px rgba(248,113,113,0.5), 0 0 90px rgba(248,113,113,0.25); }} 50% {{ box-shadow: 0 0 30px #f87171, 0 0 70px rgba(248,113,113,0.7), 0 0 120px rgba(248,113,113,0.4); }} }}

    .signal-label {{
        font-family: 'Playfair Display', serif;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: 1px;
    }}

    .label-real {{ color: #10b981; }}
    .label-fake {{ color: #ef4444; }}

    .badge {{
        margin-top: 10px;
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: {subtitle_color};
        border: 1px solid {input_border};
        padding: 5px 14px;
        border-radius: 20px;
    }}

    .conf-wrap {{ width: 260px; margin-top: 18px; }}
    .conf-track {{ width: 100%; height: 6px; background: rgba(128,128,128,0.15); border-radius: 10px; overflow: hidden; }}
    .conf-fill-real {{ height: 100%; background: linear-gradient(90deg, #10b981, #6ee7b7); border-radius: 10px; animation: fillBar 1.2s ease forwards; }}
    .conf-fill-fake {{ height: 100%; background: linear-gradient(90deg, #ef4444, #fca5a5); border-radius: 10px; animation: fillBar 1.2s ease forwards; }}
    @keyframes fillBar {{ from {{ width: 0%; }} }}
    .conf-text {{ text-align: center; margin-top: 10px; font-size: 13px; color: {subtitle_color}; letter-spacing: 0.5px; }}

    section[data-testid="stSidebar"] {{
        background: {sidebar_bg};
        border-right: 1px solid {input_border};
    }}
    section[data-testid="stSidebar"] * {{ color: {text_color} !important; }}
    div[data-testid="stMetricValue"] {{ color: #d4af37 !important; }}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🛡️ About This Project")
    st.write("This tool uses **Natural Language Processing** and **Machine Learning** (TF-IDF + Logistic Regression) to classify news content.")
    st.metric("Model Accuracy", "98.94%")
    st.markdown("---")
    st.caption("Built as a final year project")

col_a, col_b, col_c = st.columns([3, 1, 1])
with col_c:
    st.markdown('<div class="theme-toggle">', unsafe_allow_html=True)
    st.button("🌙 Dark" if not is_dark else "☀️ Light", on_click=toggle_theme, key="theme_btn")
    st.markdown('</div>', unsafe_allow_html=True)

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
