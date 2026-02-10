import cv2
import csv
import os
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================= CONFIG =================
GESTURES = {
    "1": "OK",
    "2": "HELP",
    "3": "CALL_POLICE",
    "4": "AMBULANCE",
    "5": "STOP",
    "6": "HOSTAGE",
    "7": "POSSIBLE_HOSTAGE",
    "8": "WEAPON_THREAT",
    "9": "TWO_PEOPLE_TRAPPED",
    "0": "FIRE",
    "y": "YES",
    "n": "NO",
}

SAMPLES_PER_GESTURE = 200
DATA_PATH = os.path.join(os.path.dirname(__file__), "gesture_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

# ================= MEDIAPIPE TASKS =================
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

landmarker = HandLandmarker.create_from_options(options)

# ================= CSV HEADER =================
if not os.path.exists(DATA_PATH):
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}"])
    header.append("label")

    with open(DATA_PATH, "w", newline="") as f:
        csv.writer(f).writerow(header)

# ================= COLLECT =================
def collect():
    cap = cv2.VideoCapture(0)
    timestamp = 0

    current_label = None
    collecting = False
    count = 0

    print("\n🤟 SignSOS Gesture Collector")
    print("=" * 50)
    for k, v in GESTURES.items():
        print(f"Press '{k}' → {v}")
    print("Press 'q' → Quit")
    print("=" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect_for_video(mp_image, timestamp)
        timestamp += 1

        if result.hand_landmarks and collecting:
            hand = result.hand_landmarks[0]
            row = []

            for lm in hand:
                row.extend([lm.x, lm.y])

            row.append(current_label)

            with open(DATA_PATH, "a", newline="") as f:
                csv.writer(f).writerow(row)

            count += 1
            if count >= SAMPLES_PER_GESTURE:
                collecting = False
                print(f"✅ Collected {SAMPLES_PER_GESTURE} samples for {current_label}")

        status = f"Collecting {current_label} [{count}/{SAMPLES_PER_GESTURE}]" if collecting else "Ready"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("SignSOS Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        key_char = chr(key) if key != 255 else None
        if key_char in GESTURES:
            current_label = GESTURES[key_char]
            count = 0
            collecting = True
            print(f"📸 Collecting {current_label}...")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"\n💾 Saved to {DATA_PATH}")

if __name__ == "__main__":
    collect()
