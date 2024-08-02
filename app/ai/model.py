import os
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(filepath):
    return pd.read_csv(filepath)

def load_model(model_path):
    return joblib.load(model_path)

def train_model(X, y):
    model = LogisticRegression(C=10, solver='liblinear')
    model.fit(X, y)
    return model

def test_model(X, y, model):
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)

def save_model(model, filename, dirpath = ''):
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
        
    joblib.dump(model, f"{dirpath}/{filename}")


if __name__ == "__main__":
    data = load_data("https://raw.githubusercontent.com/ktxdev/symp-check/main/backend/data/processed/symptoms_disease.csv")

    # Spliting the train and test data
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train['symptoms'], test['symptoms'], train['diseases'], test['diseases']

    # Text vectorization
    vectorizer = TfidfVectorizer(min_df=10)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    # Train the model
    model = train_model(X_train_bow, y_train)

    # Test the model
    accuracy = test_model(X_test_bow, y_test, model)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Save models
    save_model(model, "symp_check_model.pkl", "data/models")
    save_model(vectorizer, "symp_check_vectorizer.pkl", "data/models")

    