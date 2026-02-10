import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

# ========== LOAD DATA ==========
data_path = os.path.join(os.path.dirname(__file__), "gesture_data.csv")
df = pd.read_csv(data_path)

print(f"📊 Dataset shape: {df.shape}")
print(f"📋 Labels found: {df['label'].unique()}")
print(f"📋 Label counts:\n{df['label'].value_counts()}")

# ========== PREPARE FEATURES ==========
X = df.drop("label", axis=1).values
y = df["label"].values

# Normalize each sample relative to wrist (first 2 values = wrist x,y)
X_normalized = []
for row in X:
    lm = row.reshape(21, 2)
    wrist = lm[0].copy()
    lm = lm - wrist
    X_normalized.append(lm.flatten())

X_normalized = np.array(X_normalized)

# ========== SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n🏋️ Training samples: {len(X_train)}")
print(f"🧪 Testing samples: {len(X_test)}")

# ========== TRAIN ==========
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ========== EVALUATE ==========
accuracy = model.score(X_test, y_test)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
print(f"\n📊 Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

# ========== SAVE ==========
# Save model
model_path = os.path.join(os.path.dirname(__file__), "gesture_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"\n💾 Model saved to {model_path}")

# Save labels
labels = list(model.classes_)
labels_path = os.path.join(os.path.dirname(__file__), "gesture_labels.pkl")
with open(labels_path, "wb") as f:
    pickle.dump(labels, f)
print(f"💾 Labels saved to {labels_path}: {labels}")

# Also copy to API folder
api_model_path = os.path.join(os.path.dirname(__file__), "..", "signsos-api", "gesture_model.pkl")
api_labels_path = os.path.join(os.path.dirname(__file__), "..", "signsos-api", "gesture_labels.pkl")

try:
    with open(api_model_path, "wb") as f:
        pickle.dump(model, f)
    with open(api_labels_path, "wb") as f:
        pickle.dump(labels, f)
    print(f"💾 Copied to API folder")
except:
    print("⚠️ Could not copy to API folder - do it manually")

print("\n🎉 Done! Copy gesture_model.pkl and gesture_labels.pkl to signsos-api/ folder")