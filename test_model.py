
# Experiment 9: Automated tests for the machine-learning model

# Import the libraries needed for testing
import os
import joblib
from sklearn.datasets import load_iris

# Test 1: Verify that the trained model file exists
def test_model_file_exists():
    assert os.path.exists("model.joblib"), "Model file was not created"

# Test 2: Verify that the saved model can be loaded
def test_model_can_be_loaded():
    model = joblib.load("model.joblib")
    assert model is not None, "Model could not be loaded"

# Test 3: Verify that the model produces valid predictions
def test_model_predictions():
    # Load sample input data
    iris = load_iris()
    X_test = iris.data[:5]

    # Load the trained model and generate predictions
    model = joblib.load("model.joblib")
    predictions = model.predict(X_test)

    # Check the number and validity of predictions
    assert len(predictions) == 5, "Expected five predictions"
    assert all(p in [0, 1, 2] for p in predictions), \
        "Predictions contain invalid class labels"

# Run the tests when this file is executed directly
if __name__ == "__main__":
    test_model_file_exists()
    test_model_can_be_loaded()
    test_model_predictions()
    print("All 3 automated tests passed!")
