import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
import numpy as np
from scipy.special import softmax
import time
import streamlit.components.v1 as components

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Sentira CORE | Neural Sentiment Engine",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ASSETS & PREPROCESSING ---
@st.cache_resource
def load_resources():
    model = joblib.load('svc_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    return model, vectorizer

model, vectorizer = load_resources()
stop_words = set(stopwords.words('english'))

def preprocess_text(txt):
    txt = txt.lower()
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = ''.join([i for i in txt if not i.isdigit()])
    txt = txt.encode('ascii', 'ignore').decode('ascii')
    words = txt.split()
    cleaned = [w for w in words if w not in stop_words]
    return ' '.join(cleaned)

SENTIMENT_COLORS = {
    'Positive': '#00F5FF', # Cyan Neon
    'Negative': '#FF0055', # Pink Neon
    'Neutral': '#AAFF00'   # Lime Neon
}

SENTIMENT_MAP = {
    'joy': 'Positive',
    'love': 'Positive',
    'surprise': 'Positive',
    'sadness': 'Negative',
    'anger': 'Negative',
    'fear': 'Negative'
}

# --- ADVANCED CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* AGGRESSIVE CLEAN UI */
    [data-testid="stHeader"], .stAppHeader, header {
        display: none !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
        max-width: 1200px !important;
    }

    .stApp {
        background: radial-gradient(circle at 50% -20%, #1e1b4b, #020617) !important;
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hero Section */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-bottom: 60px;
    }

    .main-title {
        font-weight: 800;
        font-size: 8rem;
        background: linear-gradient(180deg, #ffffff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin: 0;
        letter-spacing: -4px;
        filter: drop-shadow(0 0 40px rgba(255,255,255,0.15));
    }

    .subtitle {
        color: #64748b;
        font-size: 1.2rem;
        letter-spacing: 8px;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 10px;
        display: block;
    }

    /* Neural Input Area */
    .stTextArea div[data-baseweb="textarea"] {
        background: #0d111b !important;
        border: 1px solid rgba(0, 245, 255, 0.1) !important;
        border-radius: 24px !important;
        transition: all 0.3s ease;
    }

    .stTextArea div[data-baseweb="textarea"]:focus-within {
        border-color: #00F5FF !important;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.1) !important;
        background: #111827 !important;
    }

    .stTextArea textarea {
        color: #fff !important;
        font-size: 1.3rem !important;
        padding: 40px !important;
        text-align: center !important;
        line-height: 1.6 !important;
    }

    /* Execute Button */
    .stButton>button {
        width: 100% !important;
        max-width: 400px;
        background: linear-gradient(90deg, #00F5FF 0%, #00D1FF 100%) !important;
        color: #000 !important;
        border-radius: 100px !important;
        padding: 20px 60px !important;
        height: auto !important;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
        border: none !important;
        display: block !important;
        margin: 40px auto 0 auto !important;
        transition: all 0.4s cubic-bezier(0.2, 1, 0.2, 1) !important;
        text-transform: uppercase;
        letter-spacing: 3px;
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.3) !important;
    }

    .stButton>button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 50px rgba(0, 245, 255, 0.5) !important;
    }

</style>
""", unsafe_allow_html=True)

# --- UI LAYOUT ---
# Hero Section
st.markdown("""
<div class="hero-container">
    <div style="font-size: 8rem; margin-bottom: 15px; filter: drop-shadow(0 0 30px rgba(0, 245, 255, 0.5));">💠</div>
    <h1 class="main-title">SENTIRA CORE</h1>
    <p class="subtitle">Neural Sentiment Processing Lab</p>
</div>
""", unsafe_allow_html=True)

# Main Input Section
col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    user_input = st.text_area("Analysis Input", placeholder="Share your thoughts to see your mood...", height=220, label_visibility="collapsed")
    
    predict_btn = st.button("EXECUTE NEURAL ANALYSIS")

    if predict_btn:
        if user_input.strip():
            # Logic
            processed = preprocess_text(user_input)
            vec = vectorizer.transform([processed])
            
            prediction = model.predict(vec)[0]
            decision_scores = model.decision_function(vec)
            probs = softmax(decision_scores, axis=1)
            conf = np.max(probs) * 100
            
            final_sentiment = SENTIMENT_MAP.get(prediction, "Neutral")
            color = SENTIMENT_COLORS.get(final_sentiment, "#fff")
            
            # --- ADVANCED RESULT COMPONENT ---
            # We use an iframe component to ensure NO CSS leaks or RAW CODE showing
            result_html = f"""
            <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;700;800&display=swap" rel="stylesheet">
            <style>
                body {{ margin: 0; background: transparent; color: #fff; font-family: 'Plus Jakarta Sans', sans-serif; }}
                .container {{
                    display: flex;
                    flex-direction: column;
                    gap: 25px;
                    animation: reveal 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);
                }}
                @keyframes reveal {{
                    from {{ opacity: 0; transform: translateY(40px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
                .main-card {{
                    background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 40px;
                    padding: 60px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 60px;
                    box-shadow: 0 40px 100px -20px rgba(0,0,0,0.5);
                }}
                .circle {{
                    width: 180px;
                    height: 180px;
                    border-radius: 50%;
                    border: 6px solid {color};
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 0 40px {color}33;
                    flex-shrink: 0;
                }}
                .conf-val {{ font-size: 4rem; font-weight: 800; line-height: 1; color: {color}; }}
                .conf-unit {{ font-size: 1rem; opacity: 0.5; font-weight: 700; }}
                .text-group {{ text-align: left; }}
                .label {{ font-size: 1rem; opacity: 0.4; letter-spacing: 4px; font-weight: 700; text-transform: uppercase; }}
                .sentiment {{ font-size: 5rem; font-weight: 800; color: {color}; line-height: 1; margin: 10px 0; text-shadow: 0 0 30px {color}44; }}
                .emotion {{ font-size: 1.4rem; color: #94a3b8; margin-top: 15px; }}
                .metric-box {{
                    background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(255,255,255,0.05);
                    border-radius: 30px;
                    padding: 40px;
                    text-align: center;
                }}
                .metric-label {{ font-size: 0.9rem; opacity: 0.4; letter-spacing: 3px; font-weight: 700; text-transform: uppercase; margin-bottom: 10px; }}
                .metric-val {{ font-size: 2.5rem; font-weight: 700; color: {color}; }}
            </style>
            <div class="container">
                <div class="main-card">
                    <div class="circle">
                        <span class="conf-val">{conf:.0f}</span>
                        <span class="conf-unit">% SCORE</span>
                    </div>
                    <div class="text-group">
                        <div class="label">NEURAL RESONANCE</div>
                        <div class="sentiment">{final_sentiment.upper()}</div>
                        <div class="emotion">Primary Attribute: <span style="color:#fff; font-weight:700;">{prediction.upper()}</span></div>
                    </div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">EMOTIONAL SPECTRUM INDEX</div>
                    <div class="metric-val">{prediction.capitalize()}</div>
                </div>
            </div>
            """
            components.html(result_html, height=550, scrolling=False)
        else:
            st.error("Input stream empty. Please provide data for analysis.")
