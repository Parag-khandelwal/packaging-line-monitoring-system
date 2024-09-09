from sklearn.ensemble import RandomForestClassifier
import joblib
from preprocess import load_and_preprocess_data
from config import MODEL_PATH, DATA_PATH

X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)

print("creating model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)

print("training model...")
model.fit(X_train, y_train)

joblib.dump(model, MODEL_PATH)
print("Model saved!")
