from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# Load your trained SVM model
try:
    model = joblib.load("gesture_model.pkl")
    print("✅ Model loaded successfully!")
    print(f"   Model classes: {model.classes_}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "classes": list(model.classes_) if model else []
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        if not data or "landmarks" not in data:
            return jsonify({"gesture": None, "error": "No landmarks provided"})
        
        landmarks = data["landmarks"]  # Should be 42 values (21 points × 2 coords)
        
        if len(landmarks) != 42:
            return jsonify({
                "gesture": None, 
                "error": f"Expected 42 values, got {len(landmarks)}"
            })
        
        if model is None:
            return jsonify({"gesture": None, "error": "Model not loaded"})
        
        # Convert to numpy array and predict
        lm_array = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(lm_array)[0]
        
        # Get confidence if available
        confidence = None
        if hasattr(model, 'decision_function'):
            try:
                decision = model.decision_function(lm_array)
                confidence = float(np.max(np.abs(decision)))
            except:
                pass
        
        print(f"🤟 Predicted: {prediction} (confidence: {confidence})")
        
        return jsonify({
            "gesture": str(prediction),
            "confidence": confidence
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"gesture": None, "error": str(e)})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == "__main__":
    print("=" * 50)
    print("🚨 SignSOS Flask API Server")
    print("=" * 50)
    print(f"🌐 Running on http://127.0.0.1:5000")
    print(f"📋 Test endpoint: http://127.0.0.1:5000/")
    print(f"🔮 Predict endpoint: POST http://127.0.0.1:5000/predict")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)