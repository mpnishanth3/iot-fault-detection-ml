#Random Forest (Production ML)

#Goal: Build a high-accuracy, robust model for fault detection 

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Input: [Temperature, Current] (training data)
X =np.array([[5, 1],[10,2],[15,3],[20,4],[25,5],[30,6]])
y=np.array([0,0,0,1,1,1])

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
