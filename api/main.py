import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import joblib
import numpy as np

# Initialize the FastAPI app
app = FastAPI(
    title="Sympcheck",
    description="Sympcheck prediction API. Visit this URL at port 8501 for the streamlit interface.",
    version="0.1.0",
)

# Define paths for model and vectorizer
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'symp_check_model.pkl')
vectorizer_path = os.path.join(current_dir, 'symp_check_vectorizer.pkl')

# Load the model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.get("/")
def home():
    return {"message": "Welcome to the Symptoms Disease Prediction API!"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    symptoms = data.get('symptoms', '')

    if not symptoms:
        return JSONResponse(content={'error': 'No symptoms provided'}, status_code=400)

    # Transform the input symptoms using the vectorizer
    symptoms_vec = vectorizer.transform([symptoms])

    # Make a prediction
    prediction = model.predict(symptoms_vec)
    
    # Return the prediction as a JSON response
    return JSONResponse(content={'prediction': prediction[0]})
