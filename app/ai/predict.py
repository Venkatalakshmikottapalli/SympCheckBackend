import joblib

model = joblib.load('/Users/ktxdev/Developer/symp-check/backend/data/models/symp_check_model.pkl')
vectorizer = joblib.load('/Users/ktxdev/Developer/symp-check/backend/data/models/symp_check_vectorizer.pkl')
    
def predict_disease(symptoms):
    query_vec = vectorizer.transform([symptoms])
    return model.predict(query_vec)