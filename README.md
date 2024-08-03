
# SympCheck

SympCheck is a health diagnostic tool that identifies potential diseases based on user-provided symptoms.

## Project Structure

SYMP_CHECK_BE_PROJECT/
├── api/
│   ├── __init__.py
│   └── main.py
├── requirements.txt
├── app/
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── ner_model.py
│   │   └── predict.py
│   └── main/
│       └── __init__.py
├── data/
│   ├── models/
│   │   ├── er_symptom_model
│   │   ├── symp_check_model.pkl
│   │   └── symp_check_vectorizer.pkl
│   └── processed/
│       ├── symptoms_disease.csv
│       └── symptoms_sentences.csv
├── notebooks/
│   ├── EDA.ipynb
│   └── model_building.ipynb
├── .gitignore
├── index.py
└── README.md


## How to Run

1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
    set the path
    $env:PYTHONPATH="$env:PYTHONPATH;."

2. Start the application:
    ```sh
    python api\main.py 
    ```

## Configuration

## Deployment

- For deployment, ensure that all dependencies are installed and the application is configured properly.

## License

This project is licensed under the MIT License.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- Virtual environment (optional but recommended)

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/SympCheck.git
   cd SympCheck

2.  Create a virtual environment and activate it:

    python -m venv env
    env\Scripts\activate  # On Windows
    source env/bin/activate  # On macOS/Linux

3.  Install the required dependencies:

    pip install -r requirements.txt

4.  Run the backend app:

    python api\main.py 

Git Commands: 

echo "# SympCheckBackend" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Venkatalakshmikottapalli/SympCheckBackend.git
git push -u origin main

git remote add origin https://github.com/Venkatalakshmikottapalli/SympCheckBackend.git
git branch -M main
git push -u origin main