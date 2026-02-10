import cv2
import mediapipe as mp
import joblib

# ----------- GOOGLE TTS IMPORTS (ADDED ONLY) -----------
from gtts import gTTS
from playsound import playsound
import os

# ---------- LOAD TRAINED MODEL ----------
model = joblib.load("gesture_model.pkl")

# ---------- GOOGLE TTS FUNCTION (ADDED ONLY) ----------
def speak_google(text):
    tts = gTTS(text=text, lang='en')
    tts.save("voice.mp3")
    playsound("voice.mp3")
    os.remove("voice.mp3")

last_spoken_gesture = ""

# ---------- MEDIAPIPE HANDS ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)

print("Camera started. Show hand gestures. Press Q to quit.")

# ---------- MAIN LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = ""

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # SAME FORMAT AS TRAINING (RAW LANDMARKS)
        lm_list = []
        for lm in hand.landmark:
            lm_list.extend([lm.x, lm.y])

        gesture = model.predict([lm_list])[0]

        # ----------- VOICE OUTPUT (ADDED ONLY) -----------
        if gesture != last_spoken_gesture:
            speak_google(gesture)
            last_spoken_gesture = gesture

        mp_draw.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS
        )

    if gesture:
        cv2.putText(
            frame,
            gesture,
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            4
        )

    cv2.imshow("Gesture Detection + Google Voice", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
