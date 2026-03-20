import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

#x has temp and current data , y has no fault and fault
data = pd.read_csv("data/sample_data.csv")

X = data[["Temp", "Current"]].values
y = data["Fault"].values

#model
model = LogisticRegression()
#train
model.fit(X,y)

#Model is learning a decision boundary w1​x1​+w2​x2​+b=0
#A line separating: Normal Fault

#prediction
test=np.array([[5,1]])
prediction = model.predict(test)
probability = model.predict_proba(test)

print("Prediction:",prediction[0])
print("Fault Probablity:",probability[0][1])

#decision
if prediction[0] == 1:
  print("Fault")
else:
  print("No Fault")
