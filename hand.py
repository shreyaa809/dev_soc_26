import cv2
import mediapipe as mp
import time
from gtts import gTTS
from playsound import playsound
import os
import threading
from collections import deque

# ---------------- AUDIO CACHE ----------------
AUDIO_DIR = "audio_cache"
os.makedirs(AUDIO_DIR, exist_ok=True)

def get_audio(text):
    filename = os.path.join(AUDIO_DIR, f"{text.replace(' ', '_')}.mp3")
    if not os.path.exists(filename):
        tts = gTTS(text=text, lang="en")
        tts.save(filename)
    return filename

# ---------------- AUDIO QUEUE (NON-BLOCKING) ----------------
audio_queue = deque()
audio_busy = False

def audio_worker():
    global audio_busy
    while True:
        if audio_queue:
            audio_busy = True
            file = audio_queue.popleft()
            playsound(file)
            audio_busy = False
        time.sleep(0.05)

threading.Thread(target=audio_worker, daemon=True).start()

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------- TIMING ----------------
HOLD_TIME = 0.25

# ---------------- STATE ----------------
current_gesture = ""
confirmed_gesture = ""
gesture_start_time = 0
last_spoken = ""

# ---------------- VOICE MAP ----------------
voice_map = {
    "HELP": "Help",
    "HOSTAGE / THREAT": "Hostage situation",
    "STOP / DANGER": "Stop. Danger",
    "FIRE": "Fire emergency",
    "AMBULANCE": "Medical emergency",
    "POLICE": "Call police",
    "YES": "Yes",
    "NO": "No",
    "OK": "Okay",
    "WAIT": "Wait"
}

# ---------------- FUNCTIONS ----------------
def get_fingers(lm):
    return [
        1 if lm[4].x < lm[3].x else 0,
        1 if lm[8].y < lm[6].y else 0,
        1 if lm[12].y < lm[10].y else 0,
        1 if lm[16].y < lm[14].y else 0,
        1 if lm[20].y < lm[18].y else 0
    ]

def thumb_inside_fist(lm):
    return lm[4].y > lm[2].y

def is_fist(f): return f == [0,0,0,0,0]
def is_palm(f): return f == [1,1,1,1,1]

def detect_gesture(f, lm):
    if is_fist(f) and thumb_inside_fist(lm):
        return "HOSTAGE / THREAT"
    if is_fist(f):
        return "HELP"
    if is_palm(f):
        return "STOP / DANGER"
    if f == [0,1,1,0,0]:
        return "AMBULANCE"
    if f == [1,0,0,0,1]:
        return "POLICE"
    if f == [0,1,0,0,1]:
        return "FIRE"
    if f == [1,0,0,0,0] and lm[4].y < lm[3].y:
        return "YES"
    if f == [1,0,0,0,0] and lm[4].y > lm[3].y:
        return "NO"
    if f == [1,1,0,0,0]:
        return "OK"
    if is_palm(f):
        return "WAIT"
    return ""

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    detected = ""
    now = time.time()

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark
            fingers = get_fingers(lm)
            detected = detect_gesture(fingers, lm)
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # ----- CONFIRMATION -----
    if detected != "" and detected != current_gesture:
        current_gesture = detected
        gesture_start_time = now

    if detected == current_gesture and detected != "":
        if now - gesture_start_time >= HOLD_TIME:
            confirmed_gesture = detected

    if detected == "":
        current_gesture = ""
        confirmed_gesture = ""
        last_spoken = ""

    # ----- QUEUE AUDIO (NON-BLOCKING) -----
    if confirmed_gesture != "" and confirmed_gesture != last_spoken:
        audio_file = get_audio(voice_map[confirmed_gesture])
        audio_queue.append(audio_file)
        last_spoken = confirmed_gesture

    # ----- DISPLAY -----
    if confirmed_gesture != "":
        emergency = [
            "HELP","STOP / DANGER","FIRE",
            "AMBULANCE","POLICE","HOSTAGE / THREAT"
        ]
        color = (0,0,255) if confirmed_gesture in emergency else (0,255,0)
        cv2.putText(frame, confirmed_gesture, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)

    cv2.imshow("Emergency & Communication System (SMOOTH)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()