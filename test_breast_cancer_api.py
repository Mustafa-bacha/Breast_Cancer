# test_breast_cancer_api.py
import requests
import json

# Sample data for breast cancer prediction
breast_cancer_data = {
    "mean radius": 14.0,
    "mean texture": 20.0,
    "mean perimeter": 90.0,
    "mean area": 600.0,
    "mean smoothness": 0.09,
    "mean compactness": 0.1,
    "mean concavity": 0.08,
    "mean concave points": 0.04,
    "mean symmetry": 0.18,
    "mean fractal dimension": 0.06,
    "radius error": 0.4,
    "texture error": 1.2,
    "perimeter error": 2.8,
    "area error": 40.0,
    "smoothness error": 0.007,
    "compactness error": 0.02,
    "concavity error": 0.03,
    "concave points error": 0.01,
    "symmetry error": 0.03,
    "fractal dimension error": 0.004,
    "worst radius": 16.0,
    "worst texture": 25.0,
    "worst perimeter": 105.0,
    "worst area": 800.0,
    "worst smoothness": 0.12,
    "worst compactness": 0.25,
    "worst concavity": 0.3,
    "worst concave points": 0.1,
    "worst symmetry": 0.3,
    "worst fractal dimension": 0.08
}

# Send a POST request to the API
response = requests.post('http://localhost:5001/predict_cancer',
                         json=breast_cancer_data,
                         headers={'Content-Type': 'application/json'})

# Print the response in a formatted way
result = response.json()
print("=== Breast Cancer Prediction Result ===")
print(f"Status: {result['status']}")
print(f"Result: {result['result']}")
print(f"Prediction (0 = Malignant, 1 = Benign): {result['prediction']}")
print(f"Probability of Benign: {(result['probability_benign'] * 100):.2f}%")
print(f"Confidence Level: {result['confidence_level']}")
print(f"Confidence Explanation: {result['confidence_explanation']}")
print(f"Timestamp: {result['timestamp']}")
print(f"Model Type: {result['model_info']['model_type']}")
print("======================================")