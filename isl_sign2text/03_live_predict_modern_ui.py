"""
Modernized Real-Time ISL Sign → Text Translator (Smooth UI Edition ✅)
---------------------------------------------------------------------
✅ Uses trained SVM + LabelEncoder
✅ Works with both hands (126 features)
✅ Smooth modern UI overlay (FPS, confidence, glow text)
✅ Debounced prediction smoothing
✅ Word formation + Voice output
"""

import cv2
import numpy as np
import mediapipe as mp
import joblib
import pyttsx3
import time
import warnings
from collections import deque
from pathlib import Path

# ===== SUPPRESS WARNINGS =====
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# ===== CONFIG =====
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "classifier.joblib"
LABEL_PATH = MODEL_DIR / "label_encoder.joblib"
SMOOTH_WINDOW = 6
SIGN_GAP = 2.2
DRAW_HAND_CONNECTIONS = True

# ===== LOAD MODEL =====
print("🔍 Loading trained model...")
clf = joblib.load(MODEL_PATH)
le = joblib.load(LABEL_PATH)
expected_features = clf.named_steps['scaler'].n_features_in_
print(f"✅ Model loaded ({expected_features} features)")
print(f"✅ Classes: {list(le.classes_)}")

# ===== TTS ENGINE =====
tts = pyttsx3.init()
tts.setProperty("rate", 175)
tts.setProperty("volume", 1.0)

# ===== MEDIAPIPE INIT =====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ===== NORMALIZE =====
def normalize_landmarks(landmarks):
    pts = landmarks.copy()
    wrist = pts[0, :2]
    pts[:, :2] -= wrist
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1)) or 1.0
    pts[:, :2] /= scale
    z = pts[:, 2]
    if np.std(z) > 1e-8:
        pts[:, 2] = (z - np.mean(z)) / np.std(z)
    return pts.reshape(-1)

# ===== UI HELPERS =====
def draw_text(img, text, pos, font_scale=1, color=(255, 255, 255), thickness=2, glow=(0, 0, 0)):
    """Draw text with glow effect."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, glow, thickness + 3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# ===== INIT CAMERA =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot access webcam")

window = deque(maxlen=SMOOTH_WINDOW)
word_buffer = []
last_sign_time = time.time()
fps_last_time = time.time()
fps = 0

print("\n🎥 Starting ISL → Text Translator")
print("Press 'q' to quit | 'c' to clear | 'b' to backspace\n")

# ===== MAIN LOOP =====
with mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pred_label, pred_conf = None, None

        if results.multi_hand_landmarks:
            if DRAW_HAND_CONNECTIONS:
                for hand in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                    )

            # Handle one or two hands
            if len(results.multi_hand_landmarks) == 2:
                lm1 = np.array([[p.x, p.y, p.z] for p in results.multi_hand_landmarks[0].landmark], np.float32)
                lm2 = np.array([[p.x, p.y, p.z] for p in results.multi_hand_landmarks[1].landmark], np.float32)
                feats = np.concatenate([normalize_landmarks(lm1), normalize_landmarks(lm2)])
            else:
                lm = np.array([[p.x, p.y, p.z] for p in results.multi_hand_landmarks[0].landmark], np.float32)
                feats = normalize_landmarks(lm)
                feats = np.concatenate([feats, np.zeros_like(feats)])  # pad

            feats = feats.reshape(1, -1)
            probs = clf.predict_proba(feats)[0]
            yhat = np.argmax(probs)
            pred_conf = np.max(probs) * 100
            pred_label = le.inverse_transform([yhat])[0]
            window.append(pred_label)
        else:
            window.append(None)

        # Stabilize
        confirmed = None
        if len(window) == window.maxlen and all(w == window[0] and w is not None for w in window):
            confirmed = window[0]

        now = time.time()

        # Update FPS
        dt = now - fps_last_time
        fps = 1.0 / dt if dt > 0 else 0
        fps_last_time = now

        # Confirm gesture
        if confirmed and (not word_buffer or word_buffer[-1] != confirmed):
            word_buffer.append(confirmed)
            print(f"🖐️ Detected: {confirmed}")
            tts.say(confirmed)
            tts.runAndWait()
            last_sign_time = now
            window.clear()

        # Word finalize
        if now - last_sign_time > SIGN_GAP and word_buffer:
            word = ''.join(word_buffer)
            print(f"🗣️ Word confirmed: {word}")
            tts.say(word)
            tts.runAndWait()
            word_buffer.clear()
            last_sign_time = now

        # ===== UI OVERLAY =====
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        draw_text(frame, f"Prediction: {pred_label or '--'}", (10, 45), 1.1, (255, 255, 255), 2)
        if pred_conf is not None:
            draw_text(frame, f"Confidence: {pred_conf:.1f}%", (10, 80), 0.8, (0, 255, 0), 2)
        draw_text(frame, "Word: " + ''.join(word_buffer), (10, 115), 1, (255, 255, 0), 2)
        draw_text(frame, f"FPS: {fps:.1f}", (530, 45), 0.8, (200, 200, 200), 1)

        cv2.imshow("🖐️ ISL Sign → Text (Smooth UI)", frame)

        # ===== Controls =====
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            word_buffer.clear()
        elif key == ord('b') and word_buffer:
            word_buffer.pop()

cap.release()
cv2.destroyAllWindows()
print("\n👋 Exiting translator. All done!")
