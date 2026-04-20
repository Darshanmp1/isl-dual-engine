"""
Flask backend for Smart ISL Translator
---------------------------------------
Endpoints:
  GET  /api/health       → health check
  POST /api/predict       → base64 JPEG → hand landmark prediction
                            ?model=svm (default) or ?model=cnn
  POST /api/tts           → text → audio/mpeg via gTTS
  POST /api/text-to-sign  → text → list of video clip filenames
  GET  /videos/<filename> → serve raw_videos MP4s
"""

import os
import io
import base64
import re
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import joblib
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from gtts import gTTS

# ─────────────────────────────────────
# Flask app setup
# ─────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────
# Paths
# ─────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
# In Docker, models are copied to /app/isl_sign2text/models
# Locally, it is ../isl_sign2text/models
MODEL_DIR = BASE_DIR / "isl_sign2text" / "models"
if not MODEL_DIR.exists():
    # Local fallback
    MODEL_DIR = BASE_DIR.parent / "isl_sign2text" / "models"

MODEL_PATH = MODEL_DIR / "classifier.joblib"
LABEL_PATH = MODEL_DIR / "label_encoder.joblib"
CNN_PATH   = MODEL_DIR / "classifier_cnn.keras"

# Determine VIDEO_DIR
VIDEO_DIR = BASE_DIR / "isl_text2sign" / "data" / "raw_videos"
if not VIDEO_DIR.exists():
    # Local fallback
    VIDEO_DIR = BASE_DIR.parent / "isl_text2sign" / "data" / "raw_videos"

# ─────────────────────────────────────
# Load ML artefacts once at startup
# ─────────────────────────────────────
# --- SVM (always required) ---
print(f"[BOOT] Loading SVM classifier: {MODEL_PATH}")
clf = joblib.load(str(MODEL_PATH))
print(f"[BOOT] Loading label encoder: {LABEL_PATH}")
le  = joblib.load(str(LABEL_PATH))
print(f"[BOOT] SVM model loaded.")

# --- CNN (optional – graceful fallback if missing) ---
cnn_model = None
if CNN_PATH.exists():
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        import tensorflow as tf
        from tensorflow import keras
        print(f"[BOOT] Loading CNN model: {CNN_PATH}")
        cnn_model = keras.models.load_model(str(CNN_PATH))
        print("[BOOT] CNN model loaded.")
    except ImportError:
        print("[BOOT] CNN load skipped: 'tensorflow' not installed. Only SVM mode available.")
    except Exception as e:
        print(f"[BOOT] CNN load failed ({type(e).__name__}): falling back to SVM.")
else:
    print(f"[BOOT] CNN model not found. Defaulting to SVM.")

# ─────────────────────────────────────
# MediaPipe Hands – reusable instance
# ─────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ─────────────────────────────────────
# Build video map (word/letter → filename)
# ─────────────────────────────────────
def _build_video_map():
    """Scan raw_videos/ and return {lowercase_key: filename} map."""
    vmap = {}
    if not VIDEO_DIR.exists():
        return vmap
    for f in os.listdir(VIDEO_DIR):
        if f.lower().endswith(".mp4"):
            key = os.path.splitext(f)[0].lower().replace("_", " ").strip()
            vmap[key] = f          # store *filename only*
    return vmap

video_map = _build_video_map()
print(f"[BOOT] Video map: {len(video_map)} entries loaded.")


# ═══════════════════════════════════════
# Helper: Landmark extraction
# ═══════════════════════════════════════

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize 21×3 hand landmarks (wrist-centered, scale-invariant, z-normalised)."""
    pts = landmarks.copy()
    wrist = pts[0, :2]
    pts[:, :2] -= wrist
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1)) or 1.0
    pts[:, :2] /= scale
    z = pts[:, 2]
    if np.std(z) > 1e-8:
        pts[:, 2] = (z - np.mean(z)) / np.std(z)
    return pts.reshape(-1)           # 63-dim


def _extract_features(bgr_image: np.ndarray):
    """
    Run MediaPipe on a BGR image and build a 126-dim feature vector.
    Returns feats (1, 126) or None if no hand found.
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    if len(results.multi_hand_landmarks) >= 2:
        lm1 = np.array(
            [[p.x, p.y, p.z] for p in results.multi_hand_landmarks[0].landmark],
            dtype=np.float32,
        )
        lm2 = np.array(
            [[p.x, p.y, p.z] for p in results.multi_hand_landmarks[1].landmark],
            dtype=np.float32,
        )
        feats = np.concatenate([normalize_landmarks(lm1), normalize_landmarks(lm2)])
    else:
        lm = np.array(
            [[p.x, p.y, p.z] for p in results.multi_hand_landmarks[0].landmark],
            dtype=np.float32,
        )
        feats = normalize_landmarks(lm)
        feats = np.concatenate([feats, np.zeros_like(feats)])   # pad second hand

    return feats.reshape(1, -1)     # (1, 126)


# ═══════════════════════════════════════
# Prediction helpers (SVM / CNN)
# ═══════════════════════════════════════

def _predict_svm(feats: np.ndarray):
    """Predict with SVM. Returns (label, confidence)."""
    yhat = clf.predict(feats)[0]
    label = le.inverse_transform([yhat])[0]

    confidence = 0.0
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(feats)[0]
        confidence = float(np.max(proba))
    elif hasattr(clf, "decision_function"):
        dec = clf.decision_function(feats)[0]
        if np.ndim(dec) == 0:
            confidence = float(1.0 / (1.0 + np.exp(-dec)))
        else:
            confidence = float(np.max(dec))
    else:
        confidence = 1.0

    return str(label), round(confidence, 4)


def _predict_cnn(feats: np.ndarray):
    """Predict with CNN. Returns (label, confidence)."""
    cnn_input = feats.reshape(1, 126, 1).astype(np.float32)
    proba = cnn_model.predict(cnn_input, verbose=0)[0]
    idx = int(np.argmax(proba))
    label = le.inverse_transform([idx])[0]
    confidence = float(proba[idx])
    return str(label), round(confidence, 4)


# ═══════════════════════════════════════
# Routes
# ═══════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "svm": True,
            "cnn": cnn_model is not None,
        }
    })


@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Return the list of gesture classes supported by the loaded model."""
    return jsonify({"classes": list(le.classes_)})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accept { "frame": "<base64 JPEG>" }, return label + confidence.
    Query param: ?model=svm (default) or ?model=cnn
    """
    # ── Determine which model to use ──
    model_choice = request.args.get("model", "svm").lower()

    data = request.get_json(force=True)
    b64 = data.get("frame", "")

    # Strip optional data-URI header
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"label": None, "confidence": 0, "error": "decode failed"}), 400
    except Exception as e:
        return jsonify({"label": None, "confidence": 0, "error": str(e)}), 400

    # ── Extract landmarks ──
    feats = _extract_features(img)
    if feats is None:
        return jsonify({"label": None, "confidence": 0.0, "model": model_choice})

    # ── Run selected model ──
    if model_choice == "cnn" and cnn_model is not None:
        label, confidence = _predict_cnn(feats)
    else:
        label, confidence = _predict_svm(feats)
        model_choice = "svm"     # in case CNN was requested but unavailable

    return jsonify({"label": label, "confidence": confidence, "model": model_choice})


@app.route("/api/tts", methods=["POST"])
def tts():
    """Accept { "text": "hello" }, return audio/mpeg binary."""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400

    try:
        speech = gTTS(text=text, lang="en", tld="co.in")
        buf = io.BytesIO()
        speech.write_to_fp(buf)
        buf.seek(0)
        return Response(buf.read(), mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/text-to-sign", methods=["POST"])
def text_to_sign():
    """
    Accept { "text": "hello world" }.
    Return { "clips": ["Hello.mp4", "W.mp4", "O.mp4", ...] }.
    """
    data = request.get_json(force=True)
    raw_text = data.get("text", "").strip()
    if not raw_text:
        return jsonify({"clips": [], "error": "empty text"}), 400

    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", raw_text).lower().strip()
    words = cleaned.split()
    clips = []

    for word in words:
        if word in video_map:
            clips.append(video_map[word])
        else:
            # spell letter-by-letter
            for ch in word:
                if ch.isalnum() and ch in video_map:
                    clips.append(video_map[ch])

    return jsonify({"clips": clips})


@app.route("/videos/<path:filename>", methods=["GET"])
def serve_video(filename):
    """Serve MP4 files from raw_videos/."""
    return send_from_directory(str(VIDEO_DIR), filename)


# ─────────────────────────────────────
# Entry point
# ─────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
