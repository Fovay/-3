# model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# Predict example
example = [[6.3, 3.3, 6.0, 2.5]]
with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)
print("Prediction:", iris.target_names[loaded_model.predict(example)[0]])