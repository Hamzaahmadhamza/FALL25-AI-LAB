import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_model_and_data(model_path="dream_model.pkl", mlb_path="mlb.joblib", dataset_path="dream_expressions_dataset.csv"):
    try:
        pipeline = joblib.load(model_path)
        mlb = joblib.load(mlb_path)
        df = pd.read_csv(dataset_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e.filename}")

    if "dream" not in df.columns or "emotion" not in df.columns:
        raise ValueError("Dataset must contain 'dream' and 'emotion' columns.")

    X = df["dream"].values
    Y = mlb.transform(df["emotion"])
    return pipeline, mlb, X, Y, df["dream"].values

def plot_confusion_matrix(Y_true, Y_pred, mlb):
    Y_true_labels = mlb.inverse_transform(Y_true)
    Y_pred_labels = mlb.inverse_transform(Y_pred)

    all_labels = sorted(set([item for sublist in Y_true_labels + Y_pred_labels for item in sublist]))
    cm = confusion_matrix(
        [all_labels.index(label) for sublist in Y_true_labels for label in sublist],
        [all_labels.index(label) for sublist in Y_pred_labels for label in sublist],
        labels=range(len(all_labels))
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=all_labels, yticklabels=all_labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def top_emotions_accuracy(Y_true_labels, Y_pred_labels):
    count = 0
    total = len(Y_true_labels)
    for true, pred in zip(Y_true_labels, Y_pred_labels):
        if pred[0] in true:
            count += 1
    return (count / total) * 100

def predict_user_dream(pipeline, mlb):
    print("\nType your dream below (type 'exit' to quit):\n")
    while True:
        dream_input = input("Your dream: ").strip()
        if dream_input.lower() == "exit":
            print("Exiting real-time prediction.")
            break
        pred = pipeline.predict([dream_input])
        emotions = mlb.inverse_transform(pred)
        print("Predicted Emotions:", ', '.join(emotions[0]), "\n")

# Load model and dataset
try:
    pipeline, mlb, X, Y, dreams = load_model_and_data()
except Exception as e:
    print("Error loading data or model:", e)
    exit()

# Evaluate on dataset
Y_pred = pipeline.predict(X)
Y_true_labels = mlb.inverse_transform(Y)
Y_pred_labels = mlb.inverse_transform(Y_pred)

accuracy = accuracy_score(Y, Y_pred)
top1_accuracy = top_emotions_accuracy(Y_true_labels, Y_pred_labels)

print("\nOverall Accuracy:", round(accuracy * 100, 2), "%")
print("Top Emotion Accuracy:", round(top1_accuracy, 2), "%\n")

# Print dataset results
print("{:<5} {:<60} {:<30} {:<30}".format("No.", "Dream", "Actual Emotions", "Predicted Emotions"))
print("-"*130)
for i, (dream, actual, pred) in enumerate(zip(dreams, Y_true_labels, Y_pred_labels), 1):
    print("{:<5} {:<60} {:<30} {:<30}".format(i, dream[:57]+"..." if len(dream)>57 else dream, ', '.join(actual), ', '.join(pred)))

# Plot confusion matrix
plot_confusion_matrix(Y, Y_pred, mlb)

# Start real-time user input prediction
predict_user_dream(pipeline, mlb)
