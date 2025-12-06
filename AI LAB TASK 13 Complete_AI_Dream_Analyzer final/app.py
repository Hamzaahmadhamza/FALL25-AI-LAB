from flask import Flask, request, render_template, jsonify
import joblib
import re
from sklearn.metrics import accuracy_score, f1_score

app = Flask(__name__)

pipeline = joblib.load("pipeline.joblib")
mlb = joblib.load("mlb.joblib")

import pandas as pd
ab = pd.read_csv("dream_expressions_dataset.csv") 

def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    dream = data.get("dream", "") if data else ""
    if not dream:
        return jsonify({"error": "No dream text provided."}), 400
    
    text = preprocess(dream)
    probs = pipeline.predict_proba([text])[0]
    labels = mlb.classes_
    
    result = [{"emotion": labels[i], "score": float(probs[i])} for i in range(len(labels))]
    result = sorted(result, key=lambda x: x["score"], reverse=True)
    top = result[:2]  
    
    return jsonify({"predictions": top, "all": result})

@app.route("/accuracy", methods=["GET"])
def get_accuracy():
    X_test = ab['dream'].tolist()
    y_test = ab['labels'].apply(lambda x: x.split(',')).tolist()  
    
    X_test_processed = [preprocess(text) for text in X_test]
    

    y_pred = pipeline.predict(X_test_processed)
    
    y_test_bin = mlb.transform(y_test)
    y_pred_bin = mlb.transform(y_pred)
    
    acc = accuracy_score(y_test_bin, y_pred_bin)
    f1 = f1_score(y_test_bin, y_pred_bin, average="micro")
    
    return jsonify({"accuracy": float(acc), "f1_score": float(f1)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

