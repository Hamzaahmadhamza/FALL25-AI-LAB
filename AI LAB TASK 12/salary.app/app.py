from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data['model']
    le = data['encoder']

@app.route("/", methods=["GET", "POST"])
def index():
    salary = None
    if request.method == "POST":
        experience = float(request.form["experience"])
        age = float(request.form["age"])
        gender = request.form["gender"]
        gender_encoded = le.transform([gender])[0]

        features = np.array([[experience, age, gender_encoded]])
        salary = model.predict(features)[0]

    return render_template("index.html", salary=salary)

if __name__ == "__main__":
    app.run(debug=True)
