from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# ================= Load Model =================
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

def preprocess_landmarks(landmarks):
    """
    landmarks: flat list of length 42 [x1,y1,x2,y2,...]
    """
    landmarks = np.array(landmarks).reshape(21, 2)

    # Normalize relative to wrist (landmark 0)
    base_x, base_y = landmarks[0]
    landmarks[:, 0] -= base_x
    landmarks[:, 1] -= base_y

    return landmarks.flatten()



# ================= Routes =================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "landmarks" not in data:
            return jsonify({"error": "Missing landmarks"}), 400

        landmarks = data["landmarks"]

        # Convert to numpy & reshape
        processed = preprocess_landmarks(landmarks)
        prediction = model.predict([processed])
        gesture = prediction[0]

        

        return jsonify({"gesture": gesture})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
