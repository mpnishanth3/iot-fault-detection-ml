import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Train model
X = np.array([
    [5,1],[10,2],[15,3],
    [20,4],[25,5],[30,6]
])

y = np.array([0,0,0,1,1,1])

model = RandomForestClassifier()
model.fit(X, y)

def predict_fault(temp, current):
    input_data = np.array([[temp, current]])
    prediction = model.predict(input_data)[0]
    
    return "FAULT" if prediction == 1 else "NORMAL"


# Test
if __name__ == "__main__":
    print(predict_fault(48, 8))
