"""This module contains functions for training a model."""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor # type: ignore
import joblib # type: ignore
import time
import structlog
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit # type: ignore
from akhil_take_home_assignment.utils import log_model_evaluation

def train_model(X: pd.DataFrame, y: pd.Series, n_estimators=100, max_depth=None,
                min_samples_split=2, min_samples_leaf=1, bootstrap=True, max_features='sqrt') -> RandomForestRegressor:
    """Train a RandomForest model with specified hyperparameters.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        n_estimators (int): Number of trees in the forest (default: 100)
        max_depth (int): Maximum depth of the trees (default: None)
        min_samples_split (int): Minimum samples required to split an internal node (default: 2)
        min_samples_leaf (int): Minimum samples required at a leaf node (default: 1)
        bootstrap (bool): Whether to bootstrap samples (default: True)
        max_features (str): Number of features to consider for the best split (default: 'sqrt')

    Returns:
        RandomForestRegressor: Trained RandomForest model
    """
    print("Starting training for Random Forest Regressor...")
    print(f"Hyperparameters - n_estimators: {n_estimators}, max_depth: {max_depth}, "
          f"min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, "
          f"bootstrap: {bootstrap}, max_features: {max_features}")

    start_time = time.time()

    # Initialize and train the Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        max_features=max_features,
        random_state=42
    )

    # Fit the model on the training data
    rf.fit(X, y)

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")

    return rf





def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   logger: structlog.stdlib.BoundLogger | None = None) -> None:
    """Evaluate the model using appropriate metrics on the test set.

    Args:
        model: Trained model
        X_test (pd.DataFrame): Test set features
        y_test (pd.Series): Test set actual target values
        logger (structlog.stdlib.BoundLogger, optional): Logger instance

    Returns:
        None
    """
    # Predicting on the test set
    print("Predicting on the test set...")
    y_test_pred = model.predict(X_test)

    # Calculate evaluation metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Print evaluation metrics
    print(f"Mean Absolute Error (MAE): {test_mae:.4f}")
    print(f"Mean Squared Error (MSE): {test_mse:.4f}")
    print(f"R-Squared (R2 Score): {test_r2:.4f}")

    # Create a dictionary of evaluation metrics
    evaluation_metrics = {
        'MAE': test_mae,
        'MSE': test_mse,
        'R2': test_r2
    }

    # Log the metrics if a logger is provided
    if logger:
        log_model_evaluation(logger, evaluation_metrics)

    return None





# Define a function for tracking progress during hyperparameter tuning
def verbose_callback(params, split, score):
    """Custom verbose callback to print progress during hyperparameter tuning.

    Args:
        params (dict): The set of hyperparameters being tested.
        split (int): The current split number in cross-validation.
        score (float): The performance score (MAE) for the current split.

    Prints:
        Progress information including the current split, hyperparameters, and the score.
    """
    print(f"Split {split + 1} - Parameters: {params}")
    print(f"Mean Absolute Error (MAE) for this split: {abs(score)}")
    print("-" * 50)





def hyperparameter_tuning(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    """Perform hyperparameter tuning for the Random Forest model using TimeSeriesSplit.

    Args:
        X (pd.DataFrame): Features, must contain 'year' and 'month' columns
        y (pd.Series): Target variable

    Returns:
        RandomForestRegressor: Model with best hyperparameters
    """
    # Sort the dataset by year and month before splitting (if not already sorted)
    X = X.sort_values(by=['year', 'month'])
    y = y.loc[X.index]

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'max_features': ['sqrt', 'log2'],
    }

    # Initialize the Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)

    # TimeSeriesSplit for cross-validation (5 splits)
    tscv = TimeSeriesSplit(n_splits=5)

    # RandomizedSearchCV for hyperparameter tuning
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=10,  # Number of parameter settings sampled
        cv=tscv,    # Use TimeSeriesSplit for cross-validation
        verbose=2,  # Set to 2 for detailed progress updates
        n_jobs=-1,  # Use all available cores
        scoring='neg_mean_absolute_error',
        return_train_score=True  # To capture training scores as well
    )

    # Fit the model on the training data
    print("Starting Random Forest hyperparameter tuning...")
    search_results = random_search.fit(X, y)

    # Extract cross-validation results
    cv_results = search_results.cv_results_
    for i in range(len(cv_results['mean_test_score'])):
        params = cv_results['params'][i]
        score = cv_results['mean_test_score'][i]
        split_num = i // 5  # Get the split number
        verbose_callback(params, split_num, score)

    # Print the best parameters and best score
    print("Hyperparameter tuning complete!")
    print("Best Parameters:", search_results.best_params_)
    print(f"Best Score (MAE): {-search_results.best_score_}")  # Convert to positive error

    return search_results.best_estimator_





def save_model(model: RandomForestRegressor, path: Path) -> None:
    """Save the model to disk.

    Args:
        model (RandomForestRegressor): Model to save
        path (Path): Path to save the model

    Returns:
        None
    """
    # Save the model using joblib
    joblib.dump(model, path)
    print(f"Model saved to {path}")

# End of Python Script
