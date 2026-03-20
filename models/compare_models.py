import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Data
data = pd.read_csv("data/sample_data.csv")

X = data[["Temp", "Current"]].values
y = data["Fault"].values

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X, y)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    
    print(f"{name}: Accuracy = {acc}")
