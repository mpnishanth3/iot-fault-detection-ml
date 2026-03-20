#Decision Trees (Explainable AI) 
#Goal: Build a model that thinks like: 

/* “IF condition → THEN decision”
IF Temp > 80 AND Current > 12 → FAULT
ELSE → NORMAL
[ ] */

import numpy as np
from sklearn.tree import DecisionTreeClassifier

[ ]

# Input: [Temperature, Current] (training data)
X =np.array([[5, 1],[10,2],[15,3],[20,4],[25,5],[30,6]])
y=np.array([0,0,1,0,1,1])

[ ]

#Model
model = DecisionTreeClassifier(max_depth=3)
#Train
model.fit(X,y)

[ ]

#Test
test = np.array([[13,5]])
prediction = model.predict(test)

print("Prediction:",prediction[0])

#decision

if prediction[0] == 1:
  print("Fault")
else:
  print("Normal")

[ ]

from sklearn.tree import export_text

tree_rules = export_text(model, feature_names=["Temp", "Current"])
print(tree_rules)

print(model.feature_importances_)
