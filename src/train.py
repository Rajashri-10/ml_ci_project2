import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

VERSION = os.getenv("GITHUB_RUN_NUMBER", "1")


data = pd.read_csv(DATA_PATH)
X = data[["feature1", "feature2"]]
y = data["label"]

model = LogisticRegression()
model.fit(X, y)


os.makedirs(MODELS_DIR, exist_ok=True)
model_path = os.path.join(MODELS_DIR, f"model_v{VERSION}.pkl")
joblib.dump(model, model_path)

print(f" Model trained and saved as {model_path}")
