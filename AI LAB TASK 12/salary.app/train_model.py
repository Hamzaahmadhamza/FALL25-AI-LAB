import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

data = pd.read_csv("dataset.csv")

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

X = data[['Experience_Years', 'Age', 'Gender']]
y = data['Salary']

model = LinearRegression()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "encoder": le}, f)

print("Model trained and saved as model.pkl")
