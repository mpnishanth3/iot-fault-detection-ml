import numpy as np
from sklearn.linear_model import LogisticRegression

#x has temp and current data , y has no fault and fault

X =np.array([[5, 1],[10,2],[15,3],[20,4],[25,5],[30,6]])
y=np.array([0,0,0,1,1,1])

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
