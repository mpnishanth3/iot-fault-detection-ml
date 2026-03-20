#Decision Trees (Explainable AI) 
#Goal: Build a model that thinks like: 

/* “IF condition → THEN decision”
IF Temp > 80 AND Current > 12 → FAULT
ELSE → NORMAL
*/

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier


# Input: [Temperature, Current] (training data)
data = pd.read_csv("data/sample_data.csv")

X = data[["Temp", "Current"]].values
y = data["Fault"].values

#Model
model = DecisionTreeClassifier(max_depth=3)
#Train
model.fit(X,y)

#Test
test = np.array([[13,5]])
prediction = model.predict(test)

print("Prediction:",prediction[0])

#decision

if prediction[0] == 1:
  print("Fault")
else:
  print("Normal")

from sklearn.tree import export_text

tree_rules = export_text(model, feature_names=["Temp", "Current"])
print(tree_rules)

print(model.feature_importances_)
