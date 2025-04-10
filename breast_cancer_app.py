# breast_cancer_app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the trained model and scaler
print("Loading breast cancer model...")
try:
    with open('breast_cancer_model.pkl', 'rb') as f:
        model_package = pickle.load(f)
except FileNotFoundError:
    print("Error: 'breast_cancer_model.pkl' not found. Please run train_breast_cancer_model.py first.")
    exit(1)

classifier = model_package['model']
scaler = model_package['scaler']
feature_names = model_package['feature_names']

@app.route('/')
def index():
    """Serve the front-end HTML page."""
    return render_template('index.html')

@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    """Handle prediction requests and return a detailed response."""
    try:
        # Get JSON data from the request
        input_json = request.get_json()

        # Convert JSON to DataFrame
        input_df = pd.DataFrame([input_json])

        # Ensure all required features are present
        prediction_df = pd.DataFrame(columns=feature_names)
        for feature in feature_names:
            if feature in input_df.columns:
                prediction_df[feature] = input_df[feature]
            else:
                return jsonify({
                    'status': 'error',
                    'message': f"Missing required feature: '{feature}'. Please provide all 30 features.",
                    'required_features': feature_names
                }), 400

        # Scale the input data
        input_scaled = scaler.transform(prediction_df)

        # Make prediction
        prediction = classifier.predict(input_scaled)[0]
        probability = classifier.predict_proba(input_scaled)[0][1]  # Probability of benign (class 1)

        # Interpret the confidence level
        confidence_level = "High" if probability > 0.9 or probability < 0.1 else "Moderate" if probability > 0.7 or probability < 0.3 else "Low"
        
        # Prepare a detailed response
        result_message = "The tumor is benign" if prediction == 1 else "The tumor is malignant"
        response = {
            'status': 'success',
            'prediction': int(prediction),
            'result': result_message,
            'probability_benign': round(float(probability), 4),
            'confidence_level': confidence_level,
            'confidence_explanation': f"The model is {confidence_level.lower()}ly confident in this prediction based on the probability score.",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': {
                'model_type': 'GradientBoostingClassifier',
                'features_used': feature_names
            }
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"An error occurred during prediction: {str(e)}",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)