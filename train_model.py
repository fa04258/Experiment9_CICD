
# Experiment 9: CI/CD Pipeline for an AI System

# Import the required libraries
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the sample Iris dataset
# The dataset contains flower measurements and their species labels
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Split the dataset into training and testing sets
# Use the same random seed to make the split reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 3: Create the machine-learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Step 4: Train the model using the training data
model.fit(X_train, y_train)

# Step 5: Evaluate the trained model on unseen test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Step 6: Display the evaluation results
print("Model training completed successfully!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"Model accuracy: {accuracy:.4f}")

# Step 7: Save the trained model for the deployment stage
# GitHub Actions can verify this file was created
joblib.dump(model, "model.joblib")
print("Model saved successfully as model.joblib")
