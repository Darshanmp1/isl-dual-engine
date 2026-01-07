import streamlit as st
import os
import sys
from pathlib import Path

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart AI ISL Translator", page_icon="🤖", layout="wide")

# =========================
# CUSTOM STYLES
# =========================
st.markdown("""
    <style>
        body {
            background-color: #000000;
        }
        .title-main {
            text-align: center;
            color: #1E90FF;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .title-sub2 {
            text-align: center;
            color: #66e0ff;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 25px;
        }
        .subtitle {
            text-align: center;
            color: #BBBBBB;
            font-size: 18px;
            font-style: italic;
            margin-bottom: 25px;
        }
        .section {
            background-color: rgba(255, 255, 255, 0.08);
            padding: 20px 25px;
            border-radius: 12px;
            color: white;
            box-shadow: 0px 3px 8px rgba(0,0,0,0.3);
        }
        .section h3 {
            color: #00BFFF;
            margin-bottom: 10px;
        }
        .section ul li {
            margin: 6px 0;
        }
        .footer {
            text-align: center;
            color: #888;
            font-size: 13px;
            margin-top: 30px;
        }

        /* 🔴 RED BUTTONS FOR BOTH */
        .red-button .stButton>button {
            background-color: #FF4B4B !important;
            color: white !important;
            border-radius: 8px;
            height: 48px;
            font-size: 17px;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# HEADER (UPDATED)
# =========================
st.markdown('<div class="title-main">🤖 Smart AI-Powered Indian Sign Language Translator</div>', unsafe_allow_html=True)

st.markdown('<div class="title-sub2">Smart AI-based Indian Sign Language Translator for Bidirectional Communication Using Machine Learning</div>',
            unsafe_allow_html=True)

st.markdown('<div class="subtitle">"Bridging the communication gap between speech and silence"</div>',
            unsafe_allow_html=True)

st.markdown("---")

# =========================
# ABOUT SECTION
# =========================
st.markdown("""
<div style="text-align:center; color:#DDDDDD; font-size:16px; margin-bottom:30px;">
This Smart ISL Translator is an <b>AI-driven bilingual communication system</b>  
designed to empower <b>hearing and speech-impaired individuals</b>.  
It combines <b>Machine Learning</b>, <b>Computer Vision</b>, and <b>Natural Language Processing</b>  
to seamlessly convert between <b>signs, text, and voice</b> — making communication inclusive and effortless.
</div>
""", unsafe_allow_html=True)

# =========================
# FEATURE SECTIONS SIDE BY SIDE
# =========================
col1, col2 = st.columns(2)

# ---- SIGN → TEXT ----
with col1:
    st.markdown("""
        <div class="section">
            <h3>🖐️ Sign → Text Translator</h3>
            <p>This module captures hand gestures through your webcam and converts them into text and voice output.</p>
            <ul>
                <li>🎥 Real-time hand gesture recognition</li>
                <li>🧠 Dual-hand detection with stability</li>
                <li>🔊 Voice output in multiple languages</li>
                <li>💬 Enhances accessibility and inclusivity</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    sign_app_path = Path("isl_sign2text") / "app" / "streamlit_app.py"
    with st.container():
        st.markdown('<div class="red-button">', unsafe_allow_html=True)
        if st.button("🚀 Launch Sign → Text", use_container_width=True):
            if not sign_app_path.exists():
                st.error(f"⚠️ Could not find {sign_app_path}")
            else:
                os.system(f"{sys.executable} -m streamlit run {sign_app_path}")
        st.markdown('</div>', unsafe_allow_html=True)

# ---- TEXT → SIGN ----
with col2:
    st.markdown("""
        <div class="section">
            <h3>✍️ Text → Sign Translator</h3>
            <p>This module converts typed English text into expressive Indian Sign Language gesture videos.</p>
            <ul>
                <li>💬 Word and letter-level video translation</li>
                <li>🎬 Smooth animation playback</li>
                <li>🌐 Bridges the gap between communities</li>
                <li>📚 Useful for education and awareness</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    text_app_path = Path("isl_text2sign") / "src" / "app.py"
    with st.container():
        st.markdown('<div class="red-button">', unsafe_allow_html=True)
        if st.button("🎬 Launch Text → Sign", use_container_width=True):
            if not text_app_path.exists():
                st.error(f"⚠️ Could not find {text_app_path}")
            else:
                os.system(f"{sys.executable} -m streamlit run {text_app_path}")
        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
  <b> </b> Enabling communication beyond barriers.
</div>
""", unsafe_allow_html=True)
