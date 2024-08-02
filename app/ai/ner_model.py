import os
import re
import spacy
import random
import pandas as pd

from spacy.training import Example
from spacy.util import minibatch, compounding

MODELS_DIR = 'data/models'
NER_MODEL_PATH = f'{MODELS_DIR}/er_symptom_model'
SYMPTOM_LABEL = "SYMPTOM"

def preprocess_symptom(symptom):
    symptom = re.sub(r'\s*\band\b\s*', ', ', symptom).strip()
    split_text = re.split(r'\s*,\s*', symptom)
    return [part for part in split_text if part]

def preprocess_data(df):
    train_data = []

    for _, row in df.iterrows():
        symptoms = preprocess_symptom(row['symptoms'])
        sentence = row['sentence']

        entities = []
        for symptom in symptoms:
            start_index = sentence.index(symptom)
            end_index = start_index + len(symptom)

            entities.append((start_index, end_index, 'SYMPTOM'))

        train_data.append((sentence, {"entities": entities}))

    return train_data

def train_ner_model(data, nlp):

    # Get the pipeline component
    ner = nlp.get_pipe("ner")

    # Add the new entity label
    ner.add_label(SYMPTOM_LABEL)

    # Disable other pipeline components during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.create_optimizer()

        for i in range(30):  # number of iterations
            random.shuffle(data)

            losses = {}
            batches = minibatch(data, size=compounding(4.0, 32.0, 1.001))

            for batch in batches:
                texts, annotations = zip(*batch)
                example = [Example.from_dict(nlp.make_doc(text), annotation) for text, annotation in zip(texts, annotations)]
                nlp.update(example, drop=0.5, losses=losses, sgd=optimizer)
                
            print("Losses", losses)

def extract_symptoms_from_text(text):
    nlp = spacy.load(NER_MODEL_PATH)
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == SYMPTOM_LABEL]


if __name__ == "__main__":
    df = pd.read_csv("https://raw.githubusercontent.com/ktxdev/symp-check/main/backend/data/processed/symptoms_sentences.csv")

    data = preprocess_data(df)

    nlp = spacy.load("en_core_web_sm")

    train_ner_model(data, nlp)

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    nlp.to_disk(NER_MODEL_PATH)