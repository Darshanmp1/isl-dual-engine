"""
ISL Sign → Text Translator (Compact Side-by-Side Layout ✅)
----------------------------------------------------------
✅ Camera feed perfectly beside instructions
✅ Compact & balanced UI
✅ Multilingual speech support
✅ Stable real-time detection
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
import tempfile
from gtts import gTTS
from googletrans import Translator
from pathlib import Path
from collections import deque
import warnings

warnings.filterwarnings("ignore")

# =========================
# 🔧 CONFIGURATION
# =========================
st.set_page_config(page_title="ISL Sign → Text Translator", layout="wide")
st.title("Indian Sign Language → Text Translator")
st.markdown("---")

# Model paths
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "classifier.joblib"
LABEL_PATH = Path(__file__).resolve().parents[1] / "models" / "label_encoder.joblib"

# =========================
# 🧠 LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    """Load the trained classifier and label encoder."""
    if not MODEL_PATH.exists() or not LABEL_PATH.exists():
        st.error("❌ Model files not found! Ensure both are in the 'models/' folder.")
        st.stop()
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_PATH)
    return clf, le


clf, le = load_model()

# =========================
# 🖐️ MEDIAPIPE INIT
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
translator = Translator()

# =========================
# 🧩 HELPER FUNCTIONS
# =========================
def normalize_landmarks(landmarks):
    """Normalize hand landmark coordinates."""
    pts = landmarks.copy()
    wrist = pts[0, :2]
    pts[:, :2] -= wrist
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1)) or 1.0
    pts[:, :2] /= scale
    z = pts[:, 2]
    if np.std(z) > 1e-8:
        pts[:, 2] = (z - np.mean(z)) / np.std(z)
    return pts.reshape(-1)


LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",
    "Tamil": "ta",
    "Telugu": "te",
}

# =========================
# 🔊 SPEECH FUNCTION
# =========================
def speak_text_streamlit(text, lang="en"):
    """Convert text to speech and play inside Streamlit."""
    if not text.strip():
        st.warning("⚠️ No text to speak.")
        return
    try:
        if lang != "en":
            try:
                text = translator.translate(text, dest=lang).text
            except Exception:
                pass
        tts = gTTS(text=text, lang=lang, tld="co.in")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            audio_path = fp.name
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3", start_time=0)
    except Exception as e:
        st.error(f"🔇 Audio Error: {e}")

# =========================
# 🎥 CAMERA FUNCTION
# =========================
def run_camera(frame_window, result_placeholder, text_placeholder):
    """Run real-time webcam translation."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot access webcam.")
        return

    st.success("✅ Translator started! Click Stop to end.")
    window = deque(maxlen=6)
    word_buffer = []
    last_sign_time = time.time()
    SIGN_GAP = 2.5
    st.session_state["current_text"] = ""

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while not st.session_state.get("stop_flag", False):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            pred_label = None

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                # Handle both hands
                if len(results.multi_hand_landmarks) == 2:
                    lm1 = np.array([[p.x, p.y, p.z] for p in results.multi_hand_landmarks[0].landmark], np.float32)
                    lm2 = np.array([[p.x, p.y, p.z] for p in results.multi_hand_landmarks[1].landmark], np.float32)
                    feats = np.concatenate([normalize_landmarks(lm1), normalize_landmarks(lm2)])
                else:
                    lm = np.array([[p.x, p.y, p.z] for p in results.multi_hand_landmarks[0].landmark], np.float32)
                    feats = normalize_landmarks(lm)
                    feats = np.concatenate([feats, np.zeros_like(feats)])  # pad second hand

                feats = feats.reshape(1, -1)
                yhat = clf.predict(feats)[0]
                pred_label = le.inverse_transform([yhat])[0]
                window.append(pred_label)
            else:
                window.append(None)

            # Stable prediction
            confirmed = None
            if len(window) == window.maxlen and all(w == window[0] and w is not None for w in window):
                confirmed = window[0]
                if not word_buffer or word_buffer[-1] != confirmed:
                    word_buffer.append(confirmed)
                    last_sign_time = time.time()
                window.clear()

            # Add space for pauses
            now = time.time()
            if now - last_sign_time > SIGN_GAP and word_buffer:
                if word_buffer[-1] != " ":
                    word_buffer.append(" ")
                last_sign_time = now

            # Display resized, bordered video
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_small = cv2.resize(frame_rgb, (900, 600))
            bordered = cv2.copyMakeBorder(frame_small, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(25, 25, 25))

            frame_window.image(bordered, channels="RGB")
            result_placeholder.markdown(f"#### 🖐️ Prediction: **{pred_label or '--'}**")
            current_text = "".join(word_buffer).strip()
            st.session_state["current_text"] = current_text
            text_placeholder.markdown(f"#### 💬 Text: **{current_text}**")

            time.sleep(0.03)

    cap.release()
    st.warning("⚠️ Translator stopped or closed.")

# =========================
# 🧭 STREAMLIT UI (SIDE-BY-SIDE)
# =========================
col1, col2 = st.columns([1.3, 1])

# Left column: camera + results
with col1:
    frame_window = st.empty()
    result_placeholder = st.empty()
    text_placeholder = st.empty()
    speak_button = st.button("🔊 Speak Text", use_container_width=True)

# Right column: instructions + control
with col2:
    selected_lang = st.selectbox("🎤 Speaking Language", list(LANGUAGES.keys()), index=0)
    tts_lang = LANGUAGES[selected_lang]

    st.markdown("### 🎯 Instructions")
    st.info(
        """
        1. Ensure your webcam is connected.  
        2. Click **Start Translator** to begin recognition.  
        3. Perform ISL letters or words clearly.  
        4. Click **Speak Text** to hear the output.  
        5. Press **Stop Translator** to end.  
        """
    )
    start_button = st.button("▶️ Start Translator", use_container_width=True)
    stop_button = st.button("⏹ Stop Translator", use_container_width=True)

# =========================
# 🎬 CONTROL LOGIC
# =========================
if start_button:
    st.session_state.stop_flag = False
    run_camera(frame_window, result_placeholder, text_placeholder)

if stop_button:
    st.session_state.stop_flag = True

if speak_button:
    text_to_speak = st.session_state.get("current_text", "")
    speak_text_streamlit(text_to_speak, LANGUAGES[selected_lang])
