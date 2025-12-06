AI-based Dream Analyzer - Project Package
========================================

Contents:
- pipeline.joblib        : trained sklearn pipeline (TfidfVectorizer + OneVsRestClassifier)
- mlb.joblib             : MultiLabelBinarizer with emotion classes
- dream_expressions_dataset.csv : synthetic dataset used for training
- app.py                 : Flask application to serve predictions
- templates/index.html   : simple front-end to paste dream text
- requirements.txt       : python dependencies

Quick start (local):
1. Create a virtual environment and install requirements:
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt

2. Run the Flask app:
   python app.py
   Then open http://localhost:5000 in your browser.

How it works:
- The project uses a TF-IDF vectorizer and a One-vs-Rest Logistic Regression classifier
  trained on a synthetic dataset of dream descriptions labelled with emotions.
- The /analyze endpoint accepts a JSON body {'dream': '...'} and returns top predicted emotions and scores.

Notes on accuracy and improvement:
- This model is a baseline trained on a small synthetic dataset for demonstration and prototyping only.
- To improve accuracy on real user dreams:
  * Collect and label a larger, diverse dataset (crowdsourcing or annotators).
  * Use a transformer-based model (e.g., fine-tune BERT) for better language understanding.
  * Add better preprocessing (negation handling, lemmatization) and augmentations.
  * Consider multi-modal inputs (images, audio) if available.
