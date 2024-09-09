import joblib
from sklearn.metrics import accuracy_score, classification_report
from preprocess import load_and_preprocess_data
from config import MODEL_PATH, DATA_PATH

X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)
model = joblib.load(MODEL_PATH)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
print(classification_report(y_test, y_pred))
