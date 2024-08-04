from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'symp_check_model.pkl')
vectorizer_path = os.path.join(current_dir, 'symp_check_vectorizer.pkl')

# Verify the paths
print(f"Current working directory: {os.getcwd()}")
print(f"Model path: {model_path}")
print(f"Vectorizer path: {vectorizer_path}")

# Check if files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

# Load the model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/')
def home():
    return "Welcome to the Symptoms Disease Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', '')
    
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400

    # Transform the input symptoms using the vectorizer
    symptoms_vec = vectorizer.transform([symptoms])

    # Make a prediction
    prediction = model.predict(symptoms_vec)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    # Remove the app.run line for production
    # app.run(host='0.0.0.0', port=8000, debug=False)
    pass  # Gunicorn will serve the app
