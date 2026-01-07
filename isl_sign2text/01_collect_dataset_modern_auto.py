"""
Auto-Capture ISL Dataset Collector (Final & Stable ✅)
------------------------------------------------------
✅ Supports letters, digits, and words
✅ Tracks both hands using MediaPipe
✅ Always shows visible green-red landmarks
✅ Auto-captures ~100 samples per label
✅ Pads single-hand data to 126 features (consistent)
✅ Appends safely to existing dataset
✅ Saves progress even if interrupted
✅ Uses static_image_mode=True for reliable detection
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import time
import traceback

# ===== CONFIG =====
DATA_DIR = Path("data_npz")
DATA_DIR.mkdir(exist_ok=True)
SAMPLES_PER_LABEL = 100
CAPTURE_DELAY = 0.03
CONFIDENCE = 0.6

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ===== NORMALIZE LANDMARKS =====
def normalize_landmarks(landmarks):
    pts = landmarks.copy()
    wrist = pts[0, :2]
    pts[:, :2] -= wrist
    dists = np.linalg.norm(pts[:, :2], axis=1)
    scale = np.max(dists) if np.max(dists) > 0 else 1.0
    pts[:, :2] /= scale
    z = pts[:, 2]
    if np.std(z) > 1e-8:
        pts[:, 2] = (z - np.mean(z)) / np.std(z)
    return pts.reshape(-1)


# ===== SAFE SAVE =====
def safe_save(mode, X, y):
    """Append new data to existing .npz safely."""
    if len(X) == 0:
        print("⚠️ No samples to save.")
        return

    save_path = DATA_DIR / f"{mode}_data_all.npz"
    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # Append if file exists
    if save_path.exists():
        try:
            existing = np.load(save_path)
            X_old, y_old = existing["X"], existing["y"]
            X = np.vstack([X_old, X])
            y = np.concatenate([y_old, y])
        except Exception as e:
            print(f"⚠️ Could not merge old data ({e}) — creating fresh file.")

    np.savez(save_path, X=X, y=y)
    print(f"\n💾 Saved total {len(y)} samples → {save_path}")


# ===== COLLECT DATA =====
def collect_samples(mode):
    if mode == "letters":
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    elif mode == "digits":
        labels = [str(i) for i in range(10)]
    elif mode == "words":
        labels = []
        print("\nEnter words (type 'done' to finish):")
        while True:
            w = input("> ").strip().upper()
            if w.lower() == "done":
                break
            if w:
                labels.append(w)
    else:
        raise ValueError("Invalid mode")

    save_path = DATA_DIR / f"{mode}_data_all.npz"
    existing_labels = set()
    if save_path.exists():
        old = np.load(save_path)
        existing_labels = set(old["y"])
        labels = [lbl for lbl in labels if lbl not in existing_labels]
        if existing_labels:
            print(f"\n⚙️ Skipping already collected: {sorted(existing_labels)}")
        print(f"🆕 Remaining to collect: {labels}")

    if not labels:
        print("✅ All gestures already collected.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot access webcam")

    collected_data, collected_labels = [], []

    try:
        with mp_hands.Hands(
            static_image_mode=True,     # ✅ Always detect every frame
            model_complexity=1,
            max_num_hands=2,
            min_detection_confidence=CONFIDENCE,
            min_tracking_confidence=CONFIDENCE
        ) as hands:
            for label in labels:
                print(f"\n👉 Ready to capture {SAMPLES_PER_LABEL} samples for '{label}'")
                print("Press [s] to start | [q] to quit\n")

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    # Draw landmarks if detected
                    hand_detected = False
                    if results.multi_hand_landmarks:
                        hand_detected = True
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                            )

                    msg = "🖐️ Hand Detected" if hand_detected else "❌ No Hand"
                    color = (0, 255, 0) if hand_detected else (0, 0, 255)
                    cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, f"Label: {label}", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, "[s] Start | [q] Quit", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    cv2.imshow("ISL Auto Data Collection", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        safe_save(mode, collected_data, collected_labels)
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                    elif key == ord('s'):
                        print(f"\n⏳ Capturing {SAMPLES_PER_LABEL} samples for '{label}'...")
                        count = 0
                        while count < SAMPLES_PER_LABEL:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame = cv2.flip(frame, 1)
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = hands.process(rgb)

                            if not results.multi_hand_landmarks:
                                continue

                            # Always output 126 features (pad single hand)
                            if len(results.multi_hand_landmarks) == 2:
                                lm1 = np.array([[p.x, p.y, p.z] for p in results.multi_hand_landmarks[0].landmark])
                                lm2 = np.array([[p.x, p.y, p.z] for p in results.multi_hand_landmarks[1].landmark])
                                feats = np.concatenate([normalize_landmarks(lm1), normalize_landmarks(lm2)])
                            else:
                                lm = np.array([[p.x, p.y, p.z] for p in results.multi_hand_landmarks[0].landmark])
                                feats = normalize_landmarks(lm)
                                feats = np.concatenate([feats, np.zeros_like(feats)])  # pad

                            collected_data.append(feats)
                            collected_labels.append(label)
                            count += 1

                            # Visual feedback
                            cv2.putText(frame, f"Capturing {count}/{SAMPLES_PER_LABEL}",
                                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            cv2.imshow("ISL Auto Data Collection", frame)
                            cv2.waitKey(1)
                            time.sleep(CAPTURE_DELAY)

                        print(f"✅ Completed {SAMPLES_PER_LABEL} samples for '{label}'")
                        break

        safe_save(mode, collected_data, collected_labels)

    except Exception as e:
        print("\n❌ Error occurred while capturing:", e)
        traceback.print_exc()
        print("\n💾 Attempting to save collected data so far...")
        safe_save(mode, collected_data, collected_labels)

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ===== MAIN =====
if __name__ == "__main__":
    print("=== ISL Auto Dataset Collector ===")
    print("Modes:\n1. letters\n2. digits\n3. words")
    mode_input = input("Select mode (letters/digits/words): ").strip().lower()
    collect_samples(mode_input)
