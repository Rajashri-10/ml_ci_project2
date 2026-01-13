import pandas as pd
import os
import joblib
from sklearn.metrics import accuracy_score

THRESHOLD = 0.80

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


model_files = sorted(
    [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
)

latest_model = model_files[-1]
model_path = os.path.join(MODELS_DIR, latest_model)


data = pd.read_csv(DATA_PATH)
X = data[["feature1", "feature2"]]
y = data["label"]


model = joblib.load(model_path)


predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)

os.makedirs(REPORTS_DIR, exist_ok=True)
with open(os.path.join(REPORTS_DIR, "metrics.txt"), "w") as f:
    f.write(f"Model: {latest_model}\n")
    f.write(f"Accuracy: {accuracy}\n")

print(f" {latest_model} Accuracy: {accuracy}")

if accuracy < THRESHOLD:
    raise Exception(" Model performance below threshold")
else:
    print("Model performance acceptable")
