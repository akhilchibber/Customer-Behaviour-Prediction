"""This file contains the fixtures that are used in the tests."""

import pytest
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor # type: ignore
from unittest import mock


@pytest.fixture
def data():
    """Return a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3],
            "product_id": ["A", "B", "A", "B", "A"],
            "date": [
                datetime(2021, 1, 1),
                datetime(2021, 1, 2),
                datetime(2021, 1, 3),
                datetime(2021, 1, 4),
                datetime(2021, 1, 5),
            ],
        }
    )





@pytest.fixture
def unclean_data():
    """Return a sample unstructured DataFrame for testing."""
    return pd.DataFrame(
        {
            "customer_id": ["XYZ", "XYZ", "XYZ", "ABC", "ABC", "DEF", "DEF", None, "MNO"],
            "product_id": ["A", "A", "B", "A", "B", "A", "A", "B", None],
            "date": [
                datetime(2021, 1, 1),
                datetime(2021, 1, 1),
                datetime(2021, 1, 2),
                datetime(2021, 1, 3),
                datetime(2021, 1, 4),
                datetime(2021, 1, 5),
                None,  # Missing value
                datetime(2021, 1, 7),  # Missing customer_id
                datetime(2021, 1, 8),  # Missing product_id
            ],
            "s.no.": [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Unnecessary column
        }
    )





@pytest.fixture
def unprocessed_data():
    """Return a sample DataFrame for testing with timezone-aware date formats."""
    return pd.DataFrame(
        {
            "customer_id": [3, 1, 2, 1, 2],
            "product_id": ["A", "B", "A", "A", "B"],
            "date": [
                "2021-01-05 10:15:30.000000+00:00",  # Latest date
                "2021-01-02 08:45:30.000000+00:00",
                "2021-01-03 13:30:00.000000+00:00",
                "2021-01-01 07:20:15.000000+00:00",  # Earliest date
                "2021-01-04 09:05:10.000000+00:00",
            ]
        }
    )





@pytest.fixture
def seasonal_data():
    """Return a sample DataFrame for testing seasonal months."""
    return pd.DataFrame({
        'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'total_transactions': [500, 800, 450, 700, 300, 950, 1100, 350, 600, 900, 400, 750]
    })





@pytest.fixture
def monthly_transaction_data():
    """Return a sample DataFrame for testing with multiple months."""
    return pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3, 1, 2, 3, 1],
            "product_id": ["A", "B", "A", "B", "A", "A", "B", "A", "B"],
            "date": [
                datetime(2021, 1, 1),
                datetime(2021, 1, 2),
                datetime(2021, 2, 3),
                datetime(2021, 2, 4),
                datetime(2021, 3, 5),
                datetime(2021, 3, 6),
                datetime(2021, 4, 7),
                datetime(2021, 4, 8),
                datetime(2021, 5, 9),
            ],
        }
    )





@pytest.fixture
def transaction_data():
    """Return a sample DataFrame for testing with multiple months and varying transactions."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5],
            "product_id": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                           "A", "A", "B", "C", "D", "E", "F", "G", "H", "I",
                           "A", "A", "B", "B", "C"],
            "date": [
                # Transactions from 2020
                datetime(2020, 1, 1), datetime(2020, 1, 15), datetime(2020, 2, 10),
                datetime(2020, 3, 5), datetime(2020, 3, 25), datetime(2020, 4, 18),
                datetime(2020, 5, 20), datetime(2020, 6, 12), datetime(2020, 7, 30),
                datetime(2020, 8, 15),
                # Transactions from the last six months (2021)
                datetime(2021, 6, 5), datetime(2021, 7, 15), datetime(2021, 8, 10),
                datetime(2021, 9, 1), datetime(2021, 9, 25), datetime(2021, 10, 10),
                datetime(2021, 10, 15), datetime(2021, 10, 20), datetime(2021, 10, 25),
                datetime(2021, 10, 30),
                # Extra transactions for variety in the last six months
                datetime(2021, 6, 1), datetime(2021, 7, 3), datetime(2021, 8, 4),
                datetime(2021, 9, 5), datetime(2021, 9, 15),
            ],
        }
    )





@pytest.fixture
def aggregated_data():
    """Return a DataFrame with customer_id, month, year, and total_transactions."""
    return pd.DataFrame({
        "customer_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "month": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "year": [2021] * 9,
        "total_transactions": [5, 3, 2, 4, 6, 1, 3, 2, 7]
    })





@pytest.fixture
def year_data():
    """Fixture to return a sample DataFrame for testing the year scaling function."""
    return pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'year': [2018, 2019, 2020, 2021, 2022],
        'total_transactions': [5, 3, 7, 2, 6]
    })





@pytest.fixture
def preprocessed_data():
    """Return a DataFrame with sample transaction data."""
    return pd.DataFrame({
        "customer_id": [8752181, 8752181, 8752181, 8752182, 8752182],
        "product_id": ['G5HEPH9A2T', 'G5HEPH9A2T', 'G5HEPH9A2T', 'G5HEPH9A2T', 'G5HEPH9A2T'],
        "date": [
            datetime(2017, 1, 1),
            datetime(2017, 2, 15),
            datetime(2017, 5, 23),
            datetime(2017, 1, 10),
            datetime(2017, 4, 8),
        ],
        "year": [2017, 2017, 2017, 2017, 2017],
        "month": [1, 2, 5, 1, 4],
    })





@pytest.fixture
def feature_engineered_data():
    """Return a DataFrame with sample transaction data."""
    return pd.DataFrame({
        "customer_id": [8752181, 8752181, 8752181, 8752181, 8752182, 8752182],
        "product_id": ['G5HEPH9A2T'] * 6,
        "date": [
            datetime(2017, 1, 1),
            datetime(2017, 2, 15),
            datetime(2017, 5, 23),
            datetime(2017, 9, 8),
            datetime(2017, 1, 10),
            datetime(2017, 4, 8)
        ],
        "year": [2017, 2017, 2017, 2017, 2017, 2017],
        "month": [1, 2, 5, 9, 1, 4],
        "total_transactions": [5, 10, 7, 3, 4, 6]
    })





@pytest.fixture
def mock_rf_model():
    """Fixture to create a mock RandomForestRegressor model with a mocked 'predict' method."""
    mock_model = mock.Mock(spec=RandomForestRegressor)

    # Mock the 'predict' method to return a predefined result
    mock_model.predict.return_value = [0.9, 0.1, 0.5]

    return mock_model





@pytest.fixture
def x_test_data():
    """Fixture to provide mock input data for predictions."""
    return pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [4.0, 5.0, 6.0]
    })

# End of Python Script
