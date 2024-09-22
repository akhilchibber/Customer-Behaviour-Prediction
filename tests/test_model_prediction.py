"""This module includes Unit tests for model load and prediction functions!"""

from sklearn.ensemble import RandomForestRegressor # type: ignore
from akhil_take_home_assignment.model_prediction import load_model, predict
from pathlib import Path
from unittest import mock
import pandas as pd

# Test function using the fixture
def test_load_model_with_fixture(mock_rf_model):
    """Test loading a RandomForestRegressor model using mock with a fixture."""
    # Mock the joblib.load function to return the mock model from the fixture
    with mock.patch('joblib.load', return_value=mock_rf_model):
        mock_path = Path("mock_model_path.joblib")  # This path is just for reference in the test

        # Load the model using the function
        loaded_model = load_model(mock_path)

        # Assertions to ensure the model was "loaded" correctly
        assert isinstance(loaded_model, RandomForestRegressor)
        assert hasattr(loaded_model, "predict")





def test_predict(mock_rf_model, x_test_data):
    """Test the predict function using a mock RandomForestRegressor."""
    # Expected output (from the mocked model's predict method)
    expected_predictions = pd.Series([0.9, 0.1, 0.5], index=x_test_data.index)

    # Call the predict function
    predictions = predict(mock_rf_model, x_test_data)

    # Assert that the predictions match the expected values
    pd.testing.assert_series_equal(predictions, expected_predictions)

    # Check that the mock model's predict method was called with the correct input
    mock_rf_model.predict.assert_called_once_with(x_test_data)

# End of Python Script
