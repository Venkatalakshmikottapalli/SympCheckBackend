import os
import joblib
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

print("Loading model and vectorizer")
try:
    model = joblib.load('api\symp_check_model.pkl')
    vectorizer = joblib.load('api\symp_check_vectorizer.pkl')
    print("Model and vectorizer loaded successfully")
except Exception as e:
    print(f"Error loading model or vectorizer: {str(e)}")
    model = None
    vectorizer = None

@app.route('/', methods=['GET', 'POST'])
def home():
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
    if model is None or vectorizer is None:
        raise Exception("Model or vectorizer not loaded")
    symptoms_vec = vectorizer.transform([symptoms])
    prediction = model.predict(symptoms_vec)
    return prediction[0]

@app.route('/predict', methods=['POST'])
def predict():
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
    info = {
        'model_type': 'Support Vector Machine (SVM)',
        'training_data': 'Dataset of medical symptoms collected from various sources. The dataset contains approximately 10,000 samples and 50 features.',
        'performance_metrics': {
            'accuracy': '0.85',  # It shoudl be replaced  with actual metrics
            'precision': '0.83', #just to show how things will look.
            'recall': '0.82',
            'f1_score': '0.82'
        }
    }
    return render_template_string("""
        <h1>Model Information</h1>
        <p><strong>Model Type:</strong> {{ info.model_type }}</p>
        <p><strong>Training Data:</strong> {{ info.training_data }}</p>
        <h2>Performance Metrics:</h2>
        <ul>
            <li>Accuracy: {{ info.performance_metrics.accuracy }}</li>
            <li>Precision: {{ info.performance_metrics.precision }}</li>
            <li>Recall: {{ info.performance_metrics.recall }}</li>
            <li>F1 Score: {{ info.performance_metrics.f1_score }}</li>
        </ul>
        <a href="/">Back to Home</a>
    """, info=info)

@app.errorhandler(404)
def page_not_found(e):
    return render_template_string("""
        <h1>404: Page Not Found</h1>
        <p>The page you requested could not be found.</p>
        <a href="/">Go to Home</a>
    """), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template_string("""
        <h1>500: Internal Server Error</h1>
        <p>An unexpected error occurred. Please try again later.</p>
        <a href="/">Go to Home</a>
    """), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)