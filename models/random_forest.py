#Random Forest (Production ML)

#Goal: Build a high-accuracy, robust model for fault detection 

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# Input: [Temperature, Current] (training data)
data = pd.read_csv("data/sample_data.csv")

X = data[["Temp", "Current"]].values
y = data["Fault"].values


#Model
model=RandomForestClassifier(n_estimators=200, random_state=42)
#Train
model.fit(X,y)

#Train
test = np.array([[18,3]])
prediction = model.predict(test)
probability = model.predict_proba(test)

print("Prediction:",prediction[0])
print("Fault Probability:",probability[0][1])

print(model.feature_importances_)
