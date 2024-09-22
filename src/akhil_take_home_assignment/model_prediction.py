"""This module contains functions for loading a model and making predictions."""

import joblib # type: ignore
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.ensemble import RandomForestRegressor # type: ignore

def load_model(path: Path) -> Any:
    """Load the model from disk.

    Args:
        path (Path): Path to the model file.

    Returns:
        Any: Model loaded from the given path.
    """
    # Load the model using joblib
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model





def predict(model: RandomForestRegressor, X: pd.DataFrame) -> pd.Series:
    """Make predictions using the model.

    Args:
        model (RandomForestRegressor): Trained model to make predictions.
        X (pd.DataFrame): Features to make predictions.

    Returns:
        pd.Series: Predictions made by the model.
    """
    # Perform predictions
    predictions = model.predict(X)
    return pd.Series(predictions, index=X.index)




def perform_prediction_and_save(model: RandomForestRegressor,
                                test_dataset: pd.DataFrame,
                                y_test: pd.Series,
                                output_file: str) -> pd.DataFrame:
    """Perform predictions using the trained model.

    This involved performing predictions,  comparing it with true labels,
    and then saving the result to a CSV.

    Args:
        model (RandomForestRegressor): Trained model to make predictions.
        test_dataset (pd.DataFrame): Test dataset containing customer_id, month,
                                     year, and all the independent variables.
        y_test (pd.Series): True labels for comparison.
        output_file (str): File path to save the resulting CSV.

    Returns:
        pd.DataFrame: Test dataset with customer_id, month, year, predictions,
                      and true labels as total_transactions.
    """
    # Store customer_id separately and retain month and year in the dataset
    customer_ids = test_dataset['customer_id'].copy()

    # Drop only customer_id before performing predictions
    X_test = test_dataset.drop(columns=['customer_id'])

    # Perform predictions using the model
    predictions = predict(model, X_test)

    # Create a DataFrame to hold the results with customer_id, month, year, total_transactions (y_test), and predictions
    result_df = pd.DataFrame({
        'customer_id': customer_ids,
        'month': test_dataset['month'],
        'year': test_dataset['year'],
        'total_transactions': y_test.values,  # Rename y_test as total_transactions
        'predictions': predictions
    })

    # Save to CSV
    result_df.to_csv(output_file, index=False)

    return result_df

# End of Python Script
