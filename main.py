"""
Real-Time Student Comprehension Detection System
KIU University - Department of Management Information System

Multi-student version: detects and tracks ALL faces simultaneously.

FIXES APPLIED:
- Bias correction weights to suppress Neutral/Happy dominance
- Lower confidence threshold so Confused/Bored are not discarded
- Second-best emotion fallback when Neutral dominates
- Histogram equalization for better face preprocessing
- Privacy consent on startup
"""

import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime
from collections import deque, defaultdict

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[INFO] TensorFlow not found – running in MOCK mode.")


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "capture_interval_sec": 5,
    "frame_width":  1280,
    "frame_height":  720,
    "face_scale_factor":  1.1,
    "face_min_neighbors": 4,
    "face_min_size": (60, 60),
    "model_input_size": (64, 64),
    "model_path": "models/resnet50_fer2013.h5",
    "log_path":   "logs/session_log.csv",

    # ── FIXED: lowered from 0.40 so Confused/Bored are not discarded ──
    "confidence_threshold": 0.25,

    "smoothing_window": 3,
    "max_students": 30,

    # ── Bias correction weights per emotion ───────────────────────────
    # FER-2013 overrepresents Neutral (~25%) and Happy (~25%).
    # Boosting underrepresented emotions helps balance predictions.
    # Values > 1.0 boost that emotion, < 1.0 suppresses it.
    "bias_weights": {
        "Angry":    1.5,
        "Disgust":  1.8,   # most underrepresented in FER-2013
        "Fear":     1.5,
        "Happy":    0.9,
        "Sad":      1.4,
        "Surprise": 1.2,
        "Neutral":  0.7,   # heavily suppressed — model defaults to this
    },

    # ── Neutral override threshold ────────────────────────────────────
    # If model says Neutral but second-best emotion scores above this,
    # use second-best instead. Catches subtle expressions.
    "neutral_override_threshold": 0.20,
}

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

EMOTION_TO_COMPREHENSION = {
    "Happy":    "Engaged",
    "Surprise": "Engaged",
    "Neutral":  "Engaged",
    "Angry":    "Confused",
    "Fear":     "Confused",
    "Disgust":  "Confused",
    "Sad":      "Bored",
}

# Color per comprehension state (BGR)
STATE_COLOR = {
    "Engaged":  (50,  205,  50),
    "Confused": (0,   165, 255),
    "Bored":    (60,   60, 220),
    "Unknown":  (128, 128, 128),
}


# ─────────────────────────────────────────────
#  MOCK CLASSIFIER
# ─────────────────────────────────────────────
class MockEmotionClassifier:
    _cycle = ["Happy", "Neutral", "Surprise", "Sad",
              "Angry", "Neutral", "Fear", "Happy", "Neutral"]
    _idx = 0

    def predict(self, x):
        label = self._cycle[self._idx % len(self._cycle)]
        self._idx += 1
        scores = np.zeros(len(EMOTION_LABELS))
        scores[EMOTION_LABELS.index(label)] = 0.88
        scores += np.random.uniform(0, 0.02, len(scores))
        return scores / scores.sum()


# ─────────────────────────────────────────────
#  FACE DETECTOR
# ─────────────────────────────────────────────
class FaceDetector:
    def __init__(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(path)

    def detect(self, gray):
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=CONFIG["face_scale_factor"],
            minNeighbors=CONFIG["face_min_neighbors"],
            minSize=CONFIG["face_min_size"],
        )
        if len(faces) == 0:
            return []
        faces = sorted(faces.tolist(), key=lambda f: f[0])
        return faces[:CONFIG["max_students"]]


# ─────────────────────────────────────────────
#  EMOTION RECOGNIZER
# ─────────────────────────────────────────────
class EmotionRecognizer:
    def __init__(self):
        if TF_AVAILABLE and os.path.exists(CONFIG["model_path"]):
            self.model = load_model(CONFIG["model_path"], compile=False)
            self.mock = False
            print("[INFO] Real model loaded:", CONFIG["model_path"])
        else:
            self.model = MockEmotionClassifier()
            self.mock = True
            reason = ("TensorFlow missing" if not TF_AVAILABLE
                      else f"Model not found at {CONFIG['model_path']}")
            print(f"[WARN] Using MOCK model ({reason})")

        # Build bias weight array in same order as EMOTION_LABELS
        self.bias = np.array([
            CONFIG["bias_weights"].get(e, 1.0) for e in EMOTION_LABELS
        ], dtype=np.float32)

    def preprocess(self, img):
        # ── Histogram equalization: improves contrast on dark/bright faces ──
        img = cv2.equalizeHist(img)
        img = cv2.resize(img, CONFIG["model_input_size"])
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)   # (64,64,1)
        img = np.expand_dims(img, axis=0)    # (1,64,64,1)
        return img

    def apply_bias(self, raw_scores):
        """
        Apply bias correction weights then renormalize.
        This suppresses Neutral/Happy dominance and boosts
        underrepresented emotions like Disgust/Fear/Sad.
        """
        corrected = raw_scores * self.bias
        corrected = corrected / corrected.sum()
        return corrected

    def pick_emotion(self, scores):
        """
        Pick final emotion label with Neutral override logic.
        If model picks Neutral but a non-Neutral emotion scores
        above the override threshold, use that instead.
        This catches subtle confused/bored expressions.
        """
        idx   = int(np.argmax(scores))
        label = EMOTION_LABELS[idx]
        conf  = float(scores[idx])

        # Neutral override: look for a stronger non-Neutral signal
        if label == "Neutral":
            non_neutral = [
                (float(scores[i]), EMOTION_LABELS[i])
                for i in range(len(EMOTION_LABELS))
                if EMOTION_LABELS[i] != "Neutral"
            ]
            non_neutral.sort(reverse=True)
            best_score, best_label = non_neutral[0]
            if best_score >= CONFIG["neutral_override_threshold"]:
                label = best_label
                conf  = best_score

        return label, conf

    def predict(self, face_gray):
        x = self.preprocess(face_gray)
        raw = (self.model.predict(x)
               if self.mock
               else self.model.predict(x, verbose=0)[0])

        corrected   = self.apply_bias(raw)
        label, conf = self.pick_emotion(corrected)
        scores_dict = {EMOTION_LABELS[i]: float(corrected[i])
                       for i in range(len(EMOTION_LABELS))}
        return label, conf, scores_dict

    def predict_batch(self, face_list):
        """
        Predict emotions for ALL students in ONE forward pass.
        Much faster than one-by-one prediction.
        """
        if not face_list:
            return []

        batch = np.stack([self.preprocess(f)[0] for f in face_list], axis=0)

        if self.mock:
            raw_list = [self.model.predict(batch[i:i+1])
                        for i in range(len(face_list))]
        else:
            raw_list = self.model.predict(batch, verbose=0)

        output = []
        for raw in raw_list:
            corrected   = self.apply_bias(raw)
            label, conf = self.pick_emotion(corrected)
            scores_dict = {EMOTION_LABELS[i]: float(corrected[i])
                           for i in range(len(EMOTION_LABELS))}
            output.append((label, conf, scores_dict))
        return output


# ─────────────────────────────────────────────
#  SESSION LOGGER
# ─────────────────────────────────────────────
class SessionLogger:
    HEADERS = (
        ["timestamp", "elapsed_sec", "student_id",
         "emotion", "confidence", "comprehension_state"]
        + [f"score_{e}" for e in EMOTION_LABELS]
    )

    def __init__(self):
        os.makedirs(os.path.dirname(CONFIG["log_path"]), exist_ok=True)
        self.file = CONFIG["log_path"]
        with open(self.file, "w", newline="") as f:
            csv.writer(f).writerow(self.HEADERS)

    def log(self, elapsed_sec, student_id, emotion, conf, state, scores_dict):
        with open(self.file, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round(elapsed_sec, 1),
                student_id,
                emotion,
                round(conf, 4),
                state,
                *[round(scores_dict.get(e, 0.0), 4) for e in EMOTION_LABELS],
            ])


# ─────────────────────────────────────────────
#  OVERLAY RENDERER
# ─────────────────────────────────────────────
def draw_overlay(frame, faces_data, session_stats, class_stats,
                 student_count, elapsed, mock_mode):
    h, w = frame.shape[:2]

    # Top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 62), (18, 18, 18), -1)
    cv2.addWeighted(bar, 0.80, frame, 0.20, 0, frame)

    title = "KIU Comprehension Monitor" + ("  [MOCK]" if mock_mode else "")
    cv2.putText(frame, title, (14, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 1, cv2.LINE_AA)
    mins, secs = int(elapsed // 60), int(elapsed % 60)
    cv2.putText(frame, f"Session {mins:02d}:{secs:02d}", (w - 200, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, (180, 180, 180), 1, cv2.LINE_AA)

    badge = f"{student_count} student{'s' if student_count != 1 else ''} detected"
    cv2.putText(frame, badge, (14, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (140, 200, 255), 1, cv2.LINE_AA)

    # Face boxes
    for (x, y, fw, fh, sid, emotion, conf, state) in faces_data:
        color = STATE_COLOR.get(state, STATE_COLOR["Unknown"])
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
        label = f"S{sid} {emotion} ({conf:.0%})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        by = max(y - 30, 65)
        cv2.rectangle(frame, (x, by), (x + tw + 8, by + th + 8), color, -1)
        cv2.putText(frame, label, (x + 4, by + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, state, (x + 4, y + fh + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    # Bottom stats panel
    panel_h = 100
    bot = frame.copy()
    cv2.rectangle(bot, (0, h - panel_h), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(bot, 0.82, frame, 0.18, 0, frame)

    total = max(sum(class_stats.values()), 1)
    entries = [
        ("Engaged",  (50,  205,  50)),
        ("Confused", (0,   165, 255)),
        ("Bored",    (60,   60, 220)),
    ]
    bw = w // 3
    for i, (state, color) in enumerate(entries):
        pct = class_stats[state] / total
        bx  = i * bw
        cv2.rectangle(frame,
                      (bx + 8, h - panel_h + 8),
                      (bx + 8 + int((bw - 20) * pct), h - panel_h + 26),
                      color, -1)
        cv2.rectangle(frame,
                      (bx + 8, h - panel_h + 8),
                      (bx + bw - 10, h - panel_h + 26),
                      (70, 70, 70), 1)
        cv2.putText(frame, f"{state}: {class_stats[state]}/{total} now",
                    (bx + 8, h - panel_h + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
        sess_pct = session_stats[state] / max(sum(session_stats.values()), 1)
        cv2.putText(frame, f"Session avg: {sess_pct:.0%}",
                    (bx + 8, h - panel_h + 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 140), 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run():
    print("\n" + "=" * 60)
    print("  KIU Comprehension Detection System — Multi-Student Mode")
    print("  Tracks ALL faces in the frame simultaneously")
    print("  Press Q to quit and save session log")
    print("=" * 60 + "\n")

    # ── Privacy consent ──────────────────────────────────────
    print("=" * 60)
    print("  PRIVACY CONSENT NOTICE")
    print("=" * 60)
    print("  - Webcam used for emotion detection only")
    print("  - No images saved — only emotion labels and scores")
    print("  - Data stored locally, anonymized by student ID")
    print("  - You may stop at any time by pressing Q")
    print("  - Participation is entirely voluntary")
    print("=" * 60)
    consent = input("\n  Type YES to consent and start: ").strip().upper()
    if consent != "YES":
        print("\n  Session cancelled. No data recorded.")
        return
    print()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    detector   = FaceDetector()
    recognizer = EmotionRecognizer()
    logger     = SessionLogger()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    student_buffers = defaultdict(lambda: deque(maxlen=CONFIG["smoothing_window"]))
    session_stats   = {"Engaged": 0, "Confused": 0, "Bored": 0}
    class_stats     = {"Engaged": 0, "Confused": 0, "Bored": 0, "Unknown": 0}

    last_check  = 0
    start       = time.time()
    last_faces  = []
    last_count  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        elapsed = time.time() - start

        if elapsed - last_check >= CONFIG["capture_interval_sec"]:
            last_check = elapsed

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detect(gray)
            last_count = len(faces)
            class_stats = {"Engaged": 0, "Confused": 0, "Bored": 0, "Unknown": 0}
            last_faces  = []

            if last_count == 0:
                print(f"[{elapsed:6.1f}s] No faces detected")
            else:
                print(f"[{elapsed:6.1f}s] {last_count} face(s) — running inference...")

                rois = [gray[y:y+fh, x:x+fw] for (x, y, fw, fh) in faces]
                predictions = recognizer.predict_batch(rois)

                for sid, ((x, y, fw, fh), (label, conf, scores)) in \
                        enumerate(zip(faces, predictions), start=1):

                    raw_state = (
                        EMOTION_TO_COMPREHENSION.get(label, "Unknown")
                        if conf >= CONFIG["confidence_threshold"]
                        else "Unknown"
                    )

                    student_buffers[sid].append(raw_state)
                    smoothed = max(
                        set(student_buffers[sid]),
                        key=list(student_buffers[sid]).count
                    )

                    class_stats[smoothed] += 1
                    if smoothed in session_stats:
                        session_stats[smoothed] += 1

                    logger.log(elapsed, sid, label, conf, smoothed, scores)
                    last_faces.append((x, y, fw, fh, sid, label, conf, smoothed))

                    print(f"         S{sid:>2}: {label:<8} ({conf:.0%}) → {smoothed}")

                e = class_stats['Engaged']
                c = class_stats['Confused']
                b = class_stats['Bored']
                print(f"         Class → Engaged:{e}  Confused:{c}  Bored:{b}")

        display = draw_overlay(
            frame.copy(), last_faces, session_stats,
            class_stats, last_count, elapsed, recognizer.mock
        )
        cv2.imshow("KIU Comprehension Monitor  |  Press Q to quit", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    total = sum(session_stats.values())
    print("\n" + "=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    for state, count in session_stats.items():
        pct = count / total * 100 if total else 0
        bar = "█" * int(pct // 5)
        print(f"  {state:<12} {count:>4} readings  ({pct:.1f}%)  {bar}")
    print(f"\n  Log saved → {CONFIG['log_path']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run()
