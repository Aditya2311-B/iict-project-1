import streamlit as st
import pickle
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime

# Set page configurations
st.set_page_config(
    page_title="The TruthGuard Times - Automated News Verifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Stopwords (matching the week1 and app.py stopwords)
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

# Text preprocessing functions
def clean_and_tokenize(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    filtered = [t for t in tokens if t not in STOPWORDS]
    return " ".join(filtered)

# Cached resources loader
@st.cache_resource
def load_classification_resources():
    vectorizer_path = "tfidf_vectorizer.pkl"
    model_names = ["KNN", "LogReg", "RandomForest", "NeuralNet"]
    vectorizer = None
    models = {}
    
    # Load TF-IDF Vectorizer
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
            
    # Load Models
    for name in model_names:
        filename = f"{name.lower()}_model.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                models[name] = pickle.load(f)
                
    return vectorizer, models

# Execute prediction logic
def perform_classification(raw_text, vectorizer, models):
    if not raw_text.strip():
        return None
        
    clean_text = clean_and_tokenize(raw_text)
    x_vec = vectorizer.transform([clean_text])
    predictions = {}
    
    for name, model in models.items():
        pred_label = model.predict(x_vec)[0]
        try:
            probs = model.predict_proba(x_vec)[0]
            # Classes order: FAKE (index 0), REAL (index 1)
            prob_fake = float(probs[0])
            prob_real = float(probs[1])
            confidence = prob_real if pred_label == 'REAL' else prob_fake
        except Exception:
            prob_fake = 0.5
            prob_real = 0.5
            confidence = 0.5
            
        predictions[name] = {
            "prediction": pred_label,
            "confidence": round(confidence * 100, 2),
            "prob_real": round(prob_real * 100, 2),
            "prob_fake": round(prob_fake * 100, 2)
        }
        
    # Aggregate consensus
    fake_votes = sum(1 for p in predictions.values() if p['prediction'] == 'FAKE')
    real_votes = sum(1 for p in predictions.values() if p['prediction'] == 'REAL')
    
    if fake_votes > real_votes:
        consensus_verdict = 'FAKE'
        consensus_confidence = sum(p['prob_fake'] for p in predictions.values()) / 4
    elif real_votes > fake_votes:
        consensus_verdict = 'REAL'
        consensus_confidence = sum(p['prob_real'] for p in predictions.values()) / 4
    else:
        # Tie break using Neural Network
        consensus_verdict = predictions['NeuralNet']['prediction']
        consensus_confidence = predictions['NeuralNet']['confidence']
        
    return {
        "predictions": predictions,
        "consensus": {
            "verdict": consensus_verdict,
            "confidence": round(consensus_confidence, 2)
        }
    }

# ------------------ STYLES & RETRO INTERFACE INITIALIZATION ------------------
# Warm newsprint theme custom CSS injector
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Lora:ital,wght@0,400..700;1,400..700&family=Inter:wght@300;400;500;600;700;900&display=swap" rel="stylesheet">

<style>
    /* Global Page Structure Styling override */
    .stApp {
        background-color: #faf9f6 !important;
        color: #121212 !important;
        font-family: 'Lora', Georgia, serif !important;
    }
    
    /* Input Elements styling (white background & dark text) */
    div[data-testid="stTextArea"] textarea {
        background-color: #ffffff !important;
        color: #121212 !important;
        border: 1px solid #c2beb6 !important;
        font-family: 'Lora', Georgia, serif !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #121212 !important;
        box-shadow: none !important;
    }
    div[data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        color: #121212 !important;
        border: 1px solid #c2beb6 !important;
        font-family: 'Inter', sans-serif !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #121212 !important;
        box-shadow: none !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: #121212 !important;
    }
    
    /* Clean layout headers & tags */
    .section-badge {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.7rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        border-bottom: 1px solid #121212 !important;
        padding-bottom: 0.25rem !important;
        margin-bottom: 0.8rem !important;
        display: block;
        color: #121212 !important;
    }

    /* Redefine button elements to look like horizontal text links */
    div[data-testid="stButton"] button {
        background-color: transparent !important;
        border: none !important;
        border-radius: 0px !important;
        color: #555555 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        padding: 0.3rem 0.5rem !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-testid="stButton"] button:hover {
        color: #121212 !important;
        background-color: #eae7e2 !important;
    }

    div[data-testid="stButton"] button:active, div[data-testid="stButton"] button:focus {
        color: #121212 !important;
        background-color: #eae7e2 !important;
        box-shadow: none !important;
    }
    
    /* Vintage credentials access pass screen styles */
    .login-card-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
        margin-bottom: 2rem;
    }
    
    .login-badge-pass {
        border: 2px solid #222;
        padding: 2rem;
        background-color: #faf9f6;
        box-shadow: 12px 12px 0px #222;
        width: 100%;
        max-width: 440px;
        position: relative;
    }

    .press-badge {
        background: #222;
        color: #faf9f6;
        display: inline-block;
        font-weight: 900;
        font-size: 0.8rem;
        letter-spacing: 0.2em;
        padding: 0.3rem 1.2rem;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    
    .press-stamp {
        position: absolute;
        top: 1.5rem;
        right: 1.5rem;
        width: 80px;
        height: 80px;
        border: 3px dashed #8c1d1d;
        border-radius: 50%;
        color: #8c1d1d;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.55rem;
        font-weight: 900;
        text-transform: uppercase;
        transform: rotate(15deg);
        text-align: center;
        line-height: 1.2;
    }

    /* Masthead styles */
    .nyt-top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        padding: 0.4rem 0;
        border-bottom: 1px solid #121212;
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .nyt-logo {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 4.8rem !important;
        line-height: 0.95;
        font-weight: 900 !important;
        letter-spacing: -0.02em;
        color: #121212;
        text-align: center;
        margin: 0.6rem 0;
        user-select: none;
    }
    
    .nyt-tagline {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 0.85rem;
        font-style: italic;
        text-align: center;
        margin-top: 0.4rem;
        color: #555555;
    }
    
    .nyt-nav-wrapper {
        border-top: 3px double #121212;
        border-bottom: 3px double #121212;
        margin: 0.4rem 0 1.8rem 0;
        padding: 0.2rem 0;
        width: 100%;
        text-align: center;
    }
    
    /* Editorial Layout Grid Elements */
    .sidebar-story-card {
        padding-bottom: 1.2rem;
        border-bottom: 1px solid #d1cfcb;
        margin-bottom: 1.2rem;
    }
    
    .sidebar-story-card:last-child {
        border-bottom: none;
    }
    
    .story-meta {
        font-family: 'Inter', sans-serif;
        font-size: 0.6rem;
        font-weight: 900;
        text-transform: uppercase;
        color: #8c1d1d;
        margin-bottom: 0.3rem;
    }
    
    .story-meta.science-tag { color: #116b3d; }
    .story-meta.tech-tag { color: #2251a3; }
    .story-meta.economy-tag { color: #802f8a; }

    .sidebar-headline {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.05rem;
        font-weight: 800;
        line-height: 1.2;
        color: #121212;
        margin-bottom: 0.3rem;
    }
    
    .sidebar-summary {
        font-size: 0.8rem;
        line-height: 1.4;
        color: #555555;
        margin-bottom: 0.4rem;
    }
    
    /* Central column story details */
    .headline-main {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        line-height: 1.15;
        color: #121212;
        margin-bottom: 0.6rem;
    }
    
    .editorial-byline {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        color: #555555;
        margin-bottom: 1.2rem;
        text-transform: uppercase;
        border-bottom: 1px solid #d1cfcb;
        padding-bottom: 0.5rem;
    }
    
    /* Results editorial extra verdict banner */
    .consensus-banner {
        border: 3px double #121212;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        background-color: #ffffff;
    }
    
    .consensus-banner.fake {
        border-color: #8c1d1d;
        background-color: #fdf5f5;
    }
    
    .consensus-banner.real {
        border-color: #116b3d;
        background-color: #f5fdf7;
    }
    
    .consensus-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 900;
        text-transform: uppercase;
        color: #555555;
        margin-bottom: 0.4rem;
    }
    
    .consensus-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.2rem;
        font-weight: 900;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
    }
    
    .consensus-banner.fake .consensus-title { color: #8c1d1d; }
    .consensus-banner.real .consensus-title { color: #116b3d; }

    .consensus-conf {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #121212;
    }
    
    /* Model sub-cards */
    .model-card {
        background-color: #ffffff;
        border: 1px solid #d1cfcb;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .model-card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.8rem;
    }
    
    .model-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
    }
    
    .verdict-badge {
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        font-weight: 900;
        padding: 0.15rem 0.5rem;
        border: 1px solid #121212;
        text-transform: uppercase;
    }
    
    .verdict-badge.fake {
        background-color: #fdf5f5;
        color: #8c1d1d;
        border-color: #8c1d1d;
    }
    
    .verdict-badge.real {
        background-color: #f5fdf7;
        color: #116b3d;
        border-color: #116b3d;
    }
    
    .meter-bar {
        height: 6px;
        background: #eeeae4;
        overflow: hidden;
        margin-bottom: 0.2rem;
    }
    
    .meter-fill {
        height: 100%;
    }
    
    .meter-fill.fake { background-color: #8c1d1d; }
    .meter-fill.real { background-color: #116b3d; }
    
    .model-split {
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        font-weight: 700;
        color: #555555;
        text-align: right;
    }
    
    /* Illustrations & static graphics framing */
    .illustration-card {
        border: 1px solid #d1cfcb;
        background: #ffffff;
        padding: 0.8rem;
        margin-bottom: 1.5rem;
    }
    
    .illustration-card h4 {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 0.9rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        border-bottom: 1px solid #d1cfcb;
        padding-bottom: 0.3rem;
    }
    
    .illustration-frame {
        background: #fbfbfa;
        width: 100%;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem;
    }
    
    .illustration-frame img {
        max-width: 100%;
        max-height: 180px;
        object-fit: contain;
    }
    
    .illustration-caption {
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        line-height: 1.3;
        color: #555555;
        margin-top: 0.5rem;
        font-style: italic;
    }

    /* Classifieds */
    .classified-item {
        background: #ffffff;
        border: 1px solid #d1cfcb;
        padding: 1.2rem;
        margin-bottom: 1rem;
        position: relative;
    }
    
    .classified-item::before {
        content: '';
        position: absolute;
        top: 4px;
        left: 4px;
        right: 4px;
        bottom: 4px;
        border: 1px dashed #eae7e2;
        pointer-events: none;
    }
    
    .classified-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #121212;
        padding-bottom: 0.4rem;
        margin-bottom: 0.8rem;
    }

    /* Override input margins inside streamlit columns */
    div[data-testid="column"] {
        padding: 0px 5px !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ SEEDS DATA FOR WIRE ARTICLES & BULLETINS ------------------
SAMPLE_STORIES = [
    {
        "id": 1,
        "category": "Politics",
        "tag_class": "story-meta",
        "wire": "Wire 1",
        "headline": "President signs Executive Order on drug pricing relief",
        "text": "President signs historic executive order to lower prescription drug costs nationwide. The package contains guidelines for Medicare to negotiate drug rates directly, representing a major policy victory for seniors."
    },
    {
        "id": 2,
        "category": "Conspiracy",
        "tag_class": "story-meta",
        "wire": "Wire 2",
        "headline": "Whistleblower exposes secret studio moon landing reels",
        "text": "BREAKING: Secret NASA document proves Earth is flat and the moon is a hologram project by government. Whistleblowers leaked raw archives demonstrating that gravity is an illusion and space flights are recorded in studio pools."
    },
    {
        "id": 3,
        "category": "Science",
        "tag_class": "story-meta science-tag",
        "wire": "Wire 3",
        "headline": "High school science club wins purification awards",
        "text": "Local high school science club wins international competition with revolutionary water purification system. The team from Westlake High constructed a lightweight, solar-powered filter using organic nanomaterials."
    },
    {
        "id": 4,
        "category": "Conspiracy",
        "tag_class": "story-meta",
        "wire": "Wire 4",
        "headline": "Archaeologists discover alien weapons inside Pyramids",
        "text": "Alien spacecraft discovered in Giza pyramids containing high-tech laser weapons. Archaeologists scanning the pyramids uncovered hidden lead-shielded compartments housing glowing metallic components dating back 10,000 years."
    },
    {
        "id": 5,
        "category": "Technology",
        "tag_class": "story-meta tech-tag",
        "wire": "Wire 5",
        "headline": "Tech giant announces stable 100-qubit processor",
        "text": "Tech giant announces breakthrough 100-qubit quantum processor with room temperature stability. The new silicon-spin architecture operates without liquid helium cooling, paving the way for commercial quantum servers."
    },
    {
        "id": 6,
        "category": "Health",
        "tag_class": "story-meta",
        "wire": "Wire 6",
        "headline": "New supplement cures aging and regenerates teeth",
        "text": "New vitamin supplement completely cures all forms of aging and regenerates teeth overnight. The developer claims the secret compounds stimulate telomere growth and allow humans to live up to 200 years without cellular decay."
    },
    {
        "id": 7,
        "category": "Economy",
        "tag_class": "story-meta economy-tag",
        "wire": "Wire 7",
        "headline": "Global carbon pact signed by 190 nations for 40% reduction",
        "text": "Global climate pact signed by 190 nations to restrict carbon emissions by 40% over the next decade. The agreement institutes binding carbon credits and investments in green energy projects for developing nations."
    },
    {
        "id": 8,
        "category": "Economy",
        "tag_class": "story-meta economy-tag",
        "wire": "Wire 8",
        "headline": "Central bank introduces negative interest stipend home loans",
        "text": "Financial crisis averted as bank introduces negative interest rate where they pay you to borrow money. The central bank announced that consumers taking home loans will receive a monthly stipend directly into checking accounts."
    }
]

# ------------------ STATE INITIALIZATION ------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Front Page"

if "draft_text_input" not in st.session_state:
    st.session_state.draft_text_input = ""

if "last_analysis_results" not in st.session_state:
    st.session_state.last_analysis_results = None

# Button load callback logic
def load_sample_into_desk(text_content):
    st.session_state.draft_text_input = text_content
    st.session_state.current_tab = "Front Page"

# ------------------ 1. AUTHENTICATION SHIELD ------------------
if not st.session_state.authenticated:
    st.markdown("<div style='height: 4rem;'></div>", unsafe_allow_html=True)
    
    # Credential clearance badge container wrapper
    st.markdown("""
    <div class="login-card-container">
        <div class="login-badge-pass">
            <div class="press-stamp">TruthGuard<br>Official<br>Clearance</div>
            <div class="press-header">
                <span class="press-badge">Press Credential Gate</span>
                <h2 style="font-size: 2.2rem; font-weight: normal; margin: 0.2rem 0;">TruthGuard</h2>
                <p style="font-family: 'Playfair Display', Georgia, serif; font-size: 0.85rem; font-style: italic; color: #555555; margin-bottom: 1rem;">
                    Secure Authentication Portal for Verification Analysts
                </p>
            </div>
    """, unsafe_allow_html=True)
    
    # Render fields using standard Streamlit form inside columns
    with st.container():
        user = st.text_input("Analyst User Name ID", value="admin")
        password = st.text_input("Security Clearance Password", type="password", value="admin123")
        
        login_submitted = st.button("Authenticate Credential", use_container_width=True)
        if login_submitted:
            if user == "admin" and password == "admin123":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.markdown("<p style='color: #8c1d1d; font-weight: 700; margin-top: 1rem;'>Invalid Credentials. Hints: admin / admin123</p>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()

# ------------------ 2. SECURED EDITORIAL WORKSPACE ------------------
# Header and Masthead
date_str = datetime.now().strftime("%A, %B %d, %Y").upper()

st.markdown(f"""<header class="nyt-header">
<div class="nyt-top-bar">
<div>{date_str}</div>
<div style="letter-spacing:0.1em; color:#555555;">REAL-TIME VERIFICATION WORKSPACE</div>
<div class="system-ticker">Consensus Acc: 92.06% ▲</div>
</div>
<div class="nyt-masthead">
<h1 class="nyt-logo">The TruthGuard Times</h1>
<p class="nyt-tagline">"All the News That's Fit to Verify - Ensemble Neural Classification Pipeline"</p>
</div>
</header>""", unsafe_allow_html=True)

# Custom dynamic tab buttons row (Newspaper Links styled)
st.markdown("<div class='nyt-nav-wrapper'>", unsafe_allow_html=True)
nav_cols = st.columns([1, 1, 1, 1.2, 3])

if nav_cols[0].button("Front Page"):
    st.session_state.current_tab = "Front Page"
if nav_cols[1].button("Analytics Desk"):
    st.session_state.current_tab = "Analytics Desk"
if nav_cols[2].button("Classifieds Bank"):
    st.session_state.current_tab = "Classifieds Bank"
if nav_cols[3].button("Live Text Profiler"):
    st.session_state.current_tab = "Live Text Profiler"
if nav_cols[4].button("Sign Out", help="Sign out current analyst clearance session"):
    st.session_state.authenticated = False
    st.session_state.current_tab = "Front Page"
    st.session_state.last_analysis_results = None
    st.session_state.draft_text_input = ""
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# Loading classification core models
vectorizer, models = load_classification_resources()

# ------------------ 3. FRONT PAGE TAB PANEL ------------------
if st.session_state.current_tab == "Front Page":
    col_left, col_center, col_right = st.columns([1.2, 2.3, 1.3])
    
    # LEFT COLUMN: WIRE STORY SELECTION
    with col_left:
        st.markdown('<span class="section-badge">Latest Wire Feeds</span>', unsafe_allow_html=True)
        
        # Render first 4 wire feed items
        for i in range(4):
            story = SAMPLE_STORIES[i]
            st.markdown(f"""
            <div class="sidebar-story-card">
                <div class="story-meta {story['tag_class']}">{story['category']} • {story['wire']}</div>
                <h4 class="sidebar-headline">{story['headline']}</h4>
                <p class="sidebar-summary">{story['text']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Action button
            if st.button("Draft for analysis", key=f"draft_{story['id']}", help="Load this wire into the editor desk"):
                load_sample_into_desk(story['text'])
                st.rerun()
                
    # CENTER COLUMN: EDITOR DESK & CLASSIFIER
    with col_center:
        st.markdown('<span class="section-badge">Verification Desk</span>', unsafe_allow_html=True)
        st.markdown('<h2 class="headline-main">Ensemble Classifier Node Evaluates Text Authenticity</h2>', unsafe_allow_html=True)
        st.markdown('<div class="editorial-byline">By TruthGuard Editorial Algorithms • Automated Security Release</div>', unsafe_allow_html=True)
        
        # Textarea editor box
        text_input = st.text_area(
            "Copy and Paste Draft Text Copy Below",
            value=st.session_state.draft_text_input,
            placeholder="Paste the body text or headline of a news article here to run classification...",
            height=250,
            label_visibility="collapsed"
        )
        
        # Keep state updated
        st.session_state.draft_text_input = text_input
        
        # Execute button
        run_btn = st.button("Execute Automated Verification", use_container_width=True)
        if run_btn:
            if not text_input.strip():
                st.warning("Please write or select a news article first.")
            else:
                with st.spinner("Extracting TF-IDF Vectors & running Neural Networks..."):
                    results = perform_classification(text_input, vectorizer, models)
                    st.session_state.last_analysis_results = results
                    
        # Render classification outputs
        results = st.session_state.last_analysis_results
        if results:
            consensus = results['consensus']
            v_class = "consensus-banner fake" if consensus['verdict'] == 'FAKE' else "consensus-banner real"
            v_title = "CONFIRMED FAKE NEWS" if consensus['verdict'] == 'FAKE' else "VERIFIED REAL NEWS"
            
            # Consensus Header Banner
            st.markdown(f"""
            <div class="{v_class}">
                <div class="consensus-label">Aggregated Consensus Verdict</div>
                <h3 class="consensus-title">{v_title}</h3>
                <div class="consensus-conf">Average Model Certainty: {consensus['confidence']}%</div>
            </div>
            <h3 class="sub-header-verdict" style="font-family: 'Playfair Display', Georgia, serif; font-size: 1.2rem; font-weight: 800; border-bottom: 1px solid #d1cfcb; padding-bottom: 0.3rem; margin-bottom: 0.8rem;">
                Ensemble Classifier Sub-Reports
            </h3>
            """, unsafe_allow_html=True)
            
            # Grid of model sub-reports
            m_col1, m_col2 = st.columns(2)
            preds = results['predictions']
            
            def render_model_card(col, prefix, display_name, acc):
                model_data = preds[prefix]
                pred_verdict = model_data['prediction']
                v_badge = "verdict-badge fake" if pred_verdict == 'FAKE' else "verdict-badge real"
                fill_class = "meter-fill fake" if pred_verdict == 'FAKE' else "meter-fill real"
                
                col.markdown(f"""
                <div class="model-card">
                    <div class="model-card-header">
                        <span class="model-title">{display_name} <span style="font-size:0.65rem; color:#555555; font-weight:normal">(Acc: {acc})</span></span>
                        <span class="{v_badge}">{pred_verdict}</span>
                    </div>
                    <div class="meter-container">
                        <div class="meter-labels">
                            <span>Confidence</span>
                            <span>{model_data['confidence']}%</span>
                        </div>
                        <div class="meter-bar">
                            <div class="{fill_class}" style="width: {model_data['confidence']}%;"></div>
                        </div>
                    </div>
                    <div class="model-split">Real: {model_data['prob_real']}% | Fake: {model_data['prob_fake']}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            render_model_card(m_col1, "KNN", "KNN Classifier", "85.79%")
            render_model_card(m_col2, "LogReg", "Logistic Reg.", "90.79%")
            render_model_card(m_col1, "RandomForest", "Random Forest", "90.40%")
            render_model_card(m_col2, "NeuralNet", "MLP Neural Net", "92.06%")
            
            # Redirect shortcut block
            st.markdown(f"""
            <div style="text-align: center; margin-top: 0.5rem; padding: 1.2rem; border: 1px solid #d1cfcb; background: #ffffff;">
                <p style="font-family: 'Playfair Display', Georgia, serif; font-size: 0.95rem; font-style: italic; margin-bottom: 0.5rem; color: #121212;">
                    Linguistic structural profiling & model probabilities are computed.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("View Live Text Profiler Desk →", use_container_width=True):
                st.session_state.current_tab = "Live Text Profiler"
                st.rerun()

    # RIGHT COLUMN: PIPELINE STATS & DIAGRAMS
    with col_right:
        st.markdown('<span class="section-badge">Analytics Graphics</span>', unsafe_allow_html=True)
        
        # Accuracies Frame
        st.markdown("""<div class="illustration-card">
            <h4>Model Accuracies</h4>
        </div>""", unsafe_allow_html=True)
        st.image("static/model_accuracy_comparison.png", use_container_width=True)
        st.markdown("""<p class="illustration-caption" style="margin-top:-0.5rem; margin-bottom:1.5rem;">
            Figure 1. Comparison of validation set accuracies across evaluated classifiers. Neural Net leads pipeline performance.
        </p>""", unsafe_allow_html=True)
        
        # Confusion Matrices Frame
        st.markdown("""<div class="illustration-card">
            <h4>Confusion Matrices</h4>
        </div>""", unsafe_allow_html=True)
        st.image("static/confusion_matrices.png", use_container_width=True)
        st.markdown("""<p class="illustration-caption" style="margin-top:-0.5rem; margin-bottom:1.5rem;">
            Figure 2. Precision-Recall balances computed during validation cycles. Indicates low margin-errors.
        </p>""", unsafe_allow_html=True)

# ------------------ 4. ANALYTICS DESK TAB PANEL ------------------
elif st.session_state.current_tab == "Analytics Desk":
    st.markdown('<span class="section-badge">Exploratory Analytics & Pipeline Statistics</span>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:\'Playfair Display\', Georgia, serif; font-size:2.2rem; font-weight:900; margin-bottom:0.3rem;">Exploratory Analytics & Pipeline Statistics</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'Inter\', sans-serif; font-size:0.85rem; font-weight:500; color:#555555; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:1.5rem;">Scientific plots computed during the preprocessing, tokenization, training, and evaluations stage</p>', unsafe_allow_html=True)
    
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.markdown("""<div class="illustration-card">
            <h4 style="font-size: 1.1rem; margin-bottom: 0.8rem;">Model Accuracy Comparison</h4>
        </div>""", unsafe_allow_html=True)
        st.image("static/model_accuracy_comparison.png", use_container_width=True)
        st.markdown("""<p class="illustration-caption" style="font-size:0.75rem; border-top:1px dashed #d1cfcb; padding-top:0.5rem; margin-top:0.5rem; margin-bottom:1.5rem;">
            Comparison scores across standard metrics. High-dimensional vectors are evaluated using four key paradigms: KNN, Logistic Regression, Random Forest ensembles, and MLP Neural Networks.
        </p>""", unsafe_allow_html=True)
        
        st.markdown("""<div class="illustration-card">
            <h4 style="font-size: 1.1rem; margin-bottom: 0.8rem;">Dataset Label Distribution</h4>
        </div>""", unsafe_allow_html=True)
        st.image("static/label_distribution.png", use_container_width=True)
        st.markdown("""<p class="illustration-caption" style="font-size:0.75rem; border-top:1px dashed #d1cfcb; padding-top:0.5rem; margin-top:0.5rem; margin-bottom:1.5rem;">
            Proportional labels inside the training corpus. Confirms a balanced dataset structure, preventing classifier bias.
        </p>""", unsafe_allow_html=True)
        
    with col_plot2:
        st.markdown("""<div class="illustration-card">
            <h4 style="font-size: 1.1rem; margin-bottom: 0.8rem;">Confusion Matrices</h4>
        </div>""", unsafe_allow_html=True)
        st.image("static/confusion_matrices.png", use_container_width=True)
        st.markdown("""<p class="illustration-caption" style="font-size:0.75rem; border-top:1px dashed #d1cfcb; padding-top:0.5rem; margin-top:0.5rem; margin-bottom:1.5rem;">
            Precision distributions across Fake and Real classifications. Diagonal coefficients represent correct predictions, verifying classifier robustness.
        </p>""", unsafe_allow_html=True)
        
        st.markdown("""<div class="illustration-card">
            <h4 style="font-size: 1.1rem; margin-bottom: 0.8rem;">Article Length Distribution</h4>
        </div>""", unsafe_allow_html=True)
        st.image("static/article_length_distribution.png", use_container_width=True)
        st.markdown("""<p class="illustration-caption" style="font-size:0.75rem; border-top:1px dashed #d1cfcb; padding-top:0.5rem; margin-top:0.5rem; margin-bottom:1.5rem;">
            Detailed distribution curve of token word counts across historical article records in standard corpora.
        </p>""", unsafe_allow_html=True)
        
    st.markdown("""<div class="illustration-card" style="width: 100%;">
        <h4 style="font-size: 1.1rem; margin-bottom: 0.8rem;">Most Common Words in Articles</h4>
    </div>""", unsafe_allow_html=True)
    st.image("static/most_common_words.png", use_container_width=True)
    st.markdown("""<p class="illustration-caption" style="font-size:0.75rem; border-top:1px dashed #d1cfcb; padding-top:0.5rem; margin-top:0.5rem;">
        Vocabulary frequency weights after custom tokenization and stopword removal cycles. Reflects prominent semantic dimensions.
    </p>""", unsafe_allow_html=True)

# ------------------ 5. CLASSIFIEDS BANK TAB PANEL ------------------
elif st.session_state.current_tab == "Classifieds Bank":
    st.markdown('<span class="section-badge">Press Bulletin Archive</span>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:\'Playfair Display\', Georgia, serif; font-size:2.2rem; font-weight:900; margin-bottom:0.3rem;">Press Bulletin Archive</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'Inter\', sans-serif; font-size:0.85rem; font-weight:500; color:#555555; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:1.5rem;">Archived wire transcripts from political, technological, economics, and health sectors</p>', unsafe_allow_html=True)
    
    # Lay out in a grid of 3 columns
    for row_idx in range(3):
        col_ads = st.columns(3)
        for col_idx in range(3):
            story_idx = row_idx * 3 + col_idx
            if story_idx < len(SAMPLE_STORIES):
                story = SAMPLE_STORIES[story_idx]
                badge_type = "fake-badge" if "Fake" in story.get("wire", "") or story["id"] in [2, 4, 6, 8] else "real-badge"
                badge_label = "Fake Wire" if badge_type == "fake-badge" else "Real Wire"
                
                with col_ads[col_idx]:
                    st.markdown(f"""
                    <div class="classified-item">
                        <div class="classified-header">
                            <span class="classified-category">{story['category']}</span>
                            <span class="classified-status {badge_type}">{badge_label}</span>
                        </div>
                        <p style="font-family: 'Lora', Georgia, serif; font-size: 0.9rem; line-height: 1.5; color: #121212; margin-bottom: 1rem;">
                            {story['text']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("Draft Story", key=f"ad_btn_{story['id']}", help="Draft this wire story into the verification desk"):
                        load_sample_into_desk(story['text'])
                        st.rerun()

# ------------------ 6. LIVE TEXT PROFILER TAB PANEL ------------------
elif st.session_state.current_tab == "Live Text Profiler":
    st.markdown('<span class="section-badge">Live Document Profiler Desk</span>', unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:\'Playfair Display\', Georgia, serif; font-size:2.2rem; font-weight:900; margin-bottom:0.3rem;">Live Document Profiler & Metrics</h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'Inter\', sans-serif; font-size:0.85rem; font-weight:500; color:#555555; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:1.5rem;">Real-time lexical distribution and model probability spreads computed on the verified text</p>', unsafe_allow_html=True)
    
    results = st.session_state.last_analysis_results
    text_draft = st.session_state.draft_text_input
    
    if not results or not text_draft.strip():
        st.markdown("""
        <div style="text-align: center; padding: 5rem 2rem; border: 1px dashed #d1cfcb; background: #ffffff;">
            <p style="font-family: 'Playfair Display', Georgia, serif; font-size: 1.5rem; font-style: italic; color: #555555; margin-bottom: 0.5rem;">
                No Text Drafted For Analysis Yet
            </p>
            <p style="font-family: 'Inter', sans-serif; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #555555;">
                Paste an article and execute verification on the Front Page to generate real-time metrics.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Compute text stats and clean tokens list
        cleaned = text_draft.lower().replace('[^a-zA-Z\s]', ' ')
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', cleaned)
        raw_tokens = cleaned.split()
        clean_tokens = [t for t in raw_tokens if len(t) > 2 and t not in STOPWORDS]
        
        # Word counts map
        freq_map = {}
        for word in clean_tokens:
            freq_map[word] = freq_map.get(word, 0) + 1
            
        sorted_words = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)[:8]
        
        word_labels = [w[0] for w in sorted_words]
        word_counts = [w[1] for w in sorted_words]
        
        # Profile metrics
        word_count = len(raw_tokens)
        char_count = len(text_draft)
        sentence_count = len(list(filter(None, re.split(r'[.!?]+', text_draft)))) or 1
        avg_word_len = round(sum(len(w) for w in raw_tokens) / word_count, 1) if word_count > 0 else 0
        est_read_time = int(np.ceil(word_count / 200))
        clean_ratio = f"{round((len(clean_tokens) / word_count) * 100, 1)}%" if word_count > 0 else "0%"
        
        # Chart elements
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("<div class='model-card'><h3>Lexical Keyword Frequency</h3></div>", unsafe_allow_html=True)
            if word_counts:
                chart_df = pd.DataFrame({
                    "Keyword": word_labels,
                    "Count": word_counts
                }).set_index("Keyword")
                st.bar_chart(chart_df, color="#121212", height=320)
            else:
                st.info("No clean keywords found.")
            st.markdown("<p class='illustration-caption'>Distribution of the top 8 cleanest keyword tokens in the analyzed copy.</p>", unsafe_allow_html=True)
            
        with chart_col2:
            st.markdown("<div class='model-card'><h3>Classifier Probability Spread</h3></div>", unsafe_allow_html=True)
            preds = results['predictions']
            
            # Map probabilities to a chart DataFrame
            models_list = ['KNN', 'Logistic Reg.', 'Random Forest', 'MLP Neural Net']
            real_probs = [preds['KNN']['prob_real'], preds['LogReg']['prob_real'], preds['RandomForest']['prob_real'], preds['NeuralNet']['prob_real']]
            fake_probs = [preds['KNN']['prob_fake'], preds['LogReg']['prob_fake'], preds['RandomForest']['prob_fake'], preds['NeuralNet']['prob_fake']]
            
            prob_df = pd.DataFrame({
                "Model": models_list,
                "Real %": real_probs,
                "Fake %": fake_probs
            }).set_index("Model")
            
            # Streamlit native chart
            st.bar_chart(prob_df, height=320, color=["#116b3d", "#8c1d1d"])
            st.markdown("<p class='illustration-caption'>Real vs Fake prediction probability percentages compared side-by-side across ensemble classifiers.</p>", unsafe_allow_html=True)
            
        # Stats table
        st.markdown(f"""
        <div class="illustration-card" style="width: 100%; margin-top: 1.5rem;">
            <h3 style="font-family: 'Playfair Display', Georgia, serif; font-size: 1.1rem; font-weight: 800; border-bottom: 1px solid #d1cfcb; padding-bottom: 0.5rem; margin-bottom: 0.8rem;">
                Dynamic Document Statistics Profile
            </h3>
            <div style="overflow-x: auto; width: 100%;">
                <table style="width: 100%; border-collapse: collapse; text-align: left; font-family: 'Inter', sans-serif; font-size: 0.85rem;">
                    <thead>
                        <tr style="border-bottom: 2px solid #121212; font-weight: 900; text-transform: uppercase; letter-spacing: 0.05em;">
                            <th style="padding: 0.7rem 0.6rem;">Metric Type</th>
                            <th style="padding: 0.7rem 0.6rem;">Computed Value</th>
                            <th style="padding: 0.7rem 0.6rem;">Metric Type</th>
                            <th style="padding: 0.7rem 0.6rem;">Computed Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid #d1cfcb;">
                            <td style="padding: 0.7rem 0.6rem; font-weight: 700;">Word Count</td>
                            <td style="padding: 0.7rem 0.6rem;">{word_count}</td>
                            <td style="padding: 0.7rem 0.6rem; font-weight: 700;">Character Count</td>
                            <td style="padding: 0.7rem 0.6rem;">{char_count}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #d1cfcb;">
                            <td style="padding: 0.7rem 0.6rem; font-weight: 700;">Sentence Count</td>
                            <td style="padding: 0.7rem 0.6rem;">{sentence_count}</td>
                            <td style="padding: 0.7rem 0.6rem; font-weight: 700;">Average Word Length</td>
                            <td style="padding: 0.7rem 0.6rem;">{avg_word_len} chars</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #d1cfcb;">
                            <td style="padding: 0.7rem 0.6rem; font-weight: 700;">Estimated Read Time</td>
                            <td style="padding: 0.7rem 0.6rem;">{est_read_time} min</td>
                            <td style="padding: 0.7rem 0.6rem; font-weight: 700;">Clean Tokens Ratio</td>
                            <td style="padding: 0.7rem 0.6rem;">{clean_ratio}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <p class="illustration-caption" style="margin-top: 1rem;">Linguistic structural distributions generated directly from raw character buffers during the manual tokenization and cleansing phase.</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("""
<footer style="border-top: 1px solid #121212; border-bottom: 1px solid #d1cfcb; padding: 1.5rem 0; margin-top: 3.5rem; text-align: center; font-family: 'Playfair Display', Georgia, serif; font-size: 0.8rem; font-style: italic;">
    <p>Copyright &copy; 2026 The TruthGuard Times. All editorial verification metrics compiled on live local networks.</p>
</footer>
""", unsafe_allow_html=True)
