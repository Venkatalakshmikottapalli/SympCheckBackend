#import the libraries
import os
import joblib
from flask import Flask, request, jsonify, render_template_string

# Set up Flask
app = Flask(__name__)

# Load model and vectorizer
print("Loading model and vectorizer")
try:
    #Assign the path for machine learning models
    model_path = os.path.join(os.path.dirname(__file__), 'symp_check_model.pkl')
    vectorizer_path = os.path.join(os.path.dirname(__file__), 'symp_check_vectorizer.pkl')
    #Load the model
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model and vectorizer loaded successfully")
except Exception as e:
    # Exception handling if modle not loaded
    print(f"Error loading model or vectorizer: {str(e)}")
    model = None
    vectorizer = None

@app.route('/', methods=['GET', 'POST'])
def home():
    #it loads the form to enter the symptoms and receives the symptoms
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        try:
            prediction = predict_disease(symptoms)
            return render_template_string("""
                <h1>Symptom-Based Disease Prediction</h1>
                <p>Symptoms: {{ symptoms }}</p>
                <p>Predicted Disease: {{ prediction }}</p>
                <a href="/">Try another prediction</a>
            """, symptoms=symptoms, prediction=prediction)
        except Exception as e:
            #Handle the exception
            return f"An error occurred: {str(e)}"

    return render_template_string("""
        <h1>Symptom-Based Disease Prediction</h1>
        <form method="post">
            <label for="symptoms">Enter symptoms (comma-separated):</label><br>
            <input type="text" id="symptoms" name="symptoms" required><br>
            <input type="submit" value="Predict">
        </form>
        <p>Example: fever, headache, cough</p>
        <h2>API Usage:</h2>
        <p>Send a POST request to /predict with JSON data: {"symptoms": "your symptoms here"}</p>
        <p>For model information, visit <a href="/model_info">/model_info</a></p>
    """)

def predict_disease(symptoms):
     # Predict disease from symptoms
    if model is None or vectorizer is None:
        raise Exception("Model or vectorizer not loaded")
    symptoms_vec = vectorizer.transform([symptoms])
    prediction = model.predict(symptoms_vec)
    return prediction[0]

@app.route('/predict', methods=['POST'])
def predict():
    # Handle API predictions
    data = request.json
    symptoms = data.get('symptoms', '')
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400
    try:
        prediction = predict_disease(symptoms)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
     # Display model information
    info = {
        'model_type': 'Logistic Regression',
        'training_data': 'The dataset includes nearly 493,890 samples of symptoms for various diseases, with each entry consisting of a query (symptoms) and a response (disease). This extensive data ensures a robust training process for accurate predictions.',
    }
    return render_template_string("""
        <h1>Model Information</h1>
        <p><strong>Model Type:</strong> {{ info.model_type }}</p>
        <p><strong>Training Data:</strong> {{ info.training_data }}</p>
        <a href="/">Back to Home</a>
    """, info=info)

@app.errorhandler(404)
def page_not_found(e):
    # Handle 404 errors
    return render_template_string("""
        <h1>404: Page Not Found</h1>
        <p>The page you requested could not be found.</p>
        <a href="/">Go to Home</a>
    """), 404

@app.errorhandler(500)
def internal_server_error(e):
     # Handle 500 errors
    return render_template_string("""
        <h1>500: Internal Server Error</h1>
        <p>An unexpected error occurred. Please try again later.</p>
        <a href="/">Go to Home</a>
    """), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
