import joblib
import numpy as np
from config import MODEL_PATH
import warnings
warnings.filterwarnings('ignore')

model = joblib.load(MODEL_PATH)

def predict_failure(data):
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    return prediction

test_cases = {
    "Test Case 1 (No Failure)": [298.1, 308.6, 1551, 42.8, 10],
    "Test Case 2 (Failure)": [320.0, 330.0, 1800, 80.0, 250],
    "Test Case 3 (Boundary)": [315.0, 325.0, 1750, 70.0, 200],
    "Test Case 4 (No Failure, High Torque)": [298.5, 308.8, 1600, 50.0, 5],
    "Test Case 5 (High Wear and Tear)": [300.0, 310.0, 1500, 45.0, 300],
    "Test Case 6 (Normal Condition)": [295.0, 305.0, 1450, 40.0, 0],
    "Test Case 7 (High Rotational Speed)": [310.0, 320.0, 2000, 85.0, 100],
}

for test_case, data in test_cases.items():
    
    prediction = predict_failure(data)

    if prediction == 0:
        print(f"{test_case}: Prediction = {prediction[0]} [NOT A FAILURE]\n")
    elif prediction == 1:
        print(f"{test_case}: Prediction = {prediction[0]} [FAILURE]\n")

