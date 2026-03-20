import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Data
X = np.array([
    [5,1],[10,2],[15,3],
    [20,4],[25,5],[30,6]
])

y = np.array([0,0,0,1,1,1])

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
