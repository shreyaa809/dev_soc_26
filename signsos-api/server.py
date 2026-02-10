from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
import traceback

app = Flask(__name__)
CORS(app)

# ========== LOAD MODEL ==========
model = None
model_labels = None

def load_model():
    global model, model_labels
    model_path = os.path.join(os.path.dirname(__file__), "gesture_model.pkl")
    labels_path = os.path.join(os.path.dirname(__file__), "gesture_labels.pkl")
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from {model_path}")
        
        # Try loading label encoder/mapping
        if os.path.exists(labels_path):
            with open(labels_path, "rb") as f:
                model_labels = pickle.load(f)
            print(f"✅ Labels loaded: {model_labels}")
        else:
            # Fallback: try to get classes from model
            if hasattr(model, 'classes_'):
                model_labels = list(model.classes_)
                print(f"✅ Labels from model.classes_: {model_labels}")
            else:
                model_labels = [
                    "OK", "HELP", "CALL_POLICE", "AMBULANCE", "STOP",
                    "HOSTAGE", "POSSIBLE_HOSTAGE", "WEAPON_THREAT",
                    "TWO_PEOPLE_TRAPPED", "FIRE", "YES", "NO"
                ]
                print(f"⚠️ Using fallback labels: {model_labels}")
                
    except Exception as e:
        print(f"❌ Model load error: {e}")
        traceback.print_exc()
        model = None

load_model()

# ========== ROUTES ==========

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "labels": model_labels if model_labels else [],
        "model_type": str(type(model).__name__) if model else None
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        landmarks = data.get("landmarks", [])
        
        if len(landmarks) != 42:
            return jsonify({"error": f"Expected 42 values, got {len(landmarks)}"}), 400
        
        # Reshape to (21, 2)
        lm_array = np.array(landmarks).reshape(21, 2)
        
        # Normalize relative to wrist (landmark 0)
        wrist = lm_array[0].copy()
        lm_array = lm_array - wrist
        
        # Flatten back to 42 features
        features = lm_array.flatten().reshape(1, -1)
        
        # Get prediction
        prediction = model.predict(features)[0]
        
        # Get confidence if available
        confidence = 0.0
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(features)[0]
                confidence = float(np.max(proba))
            except:
                confidence = 0.0
        
        # Convert prediction to string label
        if isinstance(prediction, (int, np.integer)):
            if model_labels and prediction < len(model_labels):
                gesture = str(model_labels[prediction])
            else:
                gesture = str(prediction)
        else:
            gesture = str(prediction)
        
        gesture = gesture.strip().upper().replace(" ", "_")
        
        return jsonify({
            "gesture": gesture,
            "confidence": round(confidence, 3),
            "raw_prediction": str(prediction)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/debug_model", methods=["GET"])
def debug_model():
    """Debug endpoint to check model details"""
    info = {
        "model_loaded": model is not None,
        "model_type": str(type(model).__name__) if model else None,
        "labels": model_labels,
        "has_classes": hasattr(model, 'classes_') if model else False,
        "classes": list(model.classes_) if model and hasattr(model, 'classes_') else None,
        "n_features": model.n_features_in_ if model and hasattr(model, 'n_features_in_') else None
    }
    return jsonify(info)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)