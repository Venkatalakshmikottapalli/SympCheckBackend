from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('data\models\symp_check_model.pkl')
vectorizer = joblib.load('data\models\symp_check_vectorizer.pkl')

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
    app.run(debug=True, use_reloader=False)
