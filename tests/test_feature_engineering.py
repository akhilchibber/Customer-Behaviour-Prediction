"""This module includes Unit tests for feature engineering functions!"""

from akhil_take_home_assignment.feature_engineering import add_customer_behavior_features, \
    mean_encode_customer_id, add_cyclical_month_features, add_numerical_year_feature, \
    create_smart_features, add_lag_features, add_moving_average_features, calculate_recency, \
    add_recency_frequency_features, filter_data_by_range, split_data, encode_features
import pandas as pd
from datetime import datetime

def test_add_lag_features(aggregated_data):
    """Test adding lag features to the data."""
    # Call the function to add lag features
    result = add_lag_features(aggregated_data, lag_months=3)

    # Define the expected result DataFrame
    expected_result = pd.DataFrame({
        "customer_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "month": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "year": [2021] * 9,
        "total_transactions": [5, 3, 2, 4, 6, 1, 3, 2, 7],
        "transactions_lag_1": [0, 5, 3, 0, 4, 6, 0, 3, 2],
        "transactions_lag_2": [0, 0, 5, 0, 0, 4, 0, 0, 3],
        "transactions_lag_3": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    })

    # Assert that the result matches the expected result
    pd.testing.assert_frame_equal(result, expected_result)





def test_add_moving_average_features(aggregated_data):
    """Test the function that adds moving average features."""
    # Call the function to add moving average features
    result = add_moving_average_features(aggregated_data)

    # Define the expected result with moving average columns added
    expected_result = pd.DataFrame({
        "customer_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "month": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "year": [2021] * 9,
        "total_transactions": [5, 3, 2, 4, 6, 1, 3, 2, 7],
        "moving_avg_3": [5.0, 4.0, 3.33, 4.0, 5.0, 3.67, 3.0, 2.5, 4.0],
        "moving_avg_6": [5.0, 4.0, 3.33, 4.0, 5.0, 3.67, 3.0, 2.5, 4.0],
    })

    # Assert that the result matches the expected result
    pd.testing.assert_frame_equal(result, expected_result, rtol=1e-2)





def test_calculate_recency(aggregated_data):
    """Test the function that calculates recency for each customer."""
    # Call the function to calculate recency
    result = calculate_recency(aggregated_data)

    # Define the expected result DataFrame with the 'recency' column added
    expected_result = pd.DataFrame({
        "customer_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "month": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "year": [2021] * 9,
        "total_transactions": [5, 3, 2, 4, 6, 1, 3, 2, 7],
        "recency": [0, 1, 1, 0, 1, 1, 0, 1, 1],
    })

    # Assert that the result matches the expected result
    pd.testing.assert_frame_equal(result, expected_result)





def test_add_recency_frequency_features(aggregated_data):
    """Test the function that adds recency and frequency features."""
    # Call the function to add recency and frequency features
    result = add_recency_frequency_features(aggregated_data)

    # Define the expected result DataFrame with the 'recency' and 'frequency' columns added
    expected_result = pd.DataFrame({
        "customer_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "month": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "year": [2021] * 9,
        "total_transactions": [5, 3, 2, 4, 6, 1, 3, 2, 7],
        "recency": [0, 1, 1, 0, 1, 1, 0, 1, 1],
        "frequency": [3.33, 3.33, 3.33, 3.67, 3.67, 3.67, 4.0, 4.0, 4.0],
    })

    # Assert that the result matches the expected result
    pd.testing.assert_frame_equal(result, expected_result, rtol=1e-2)





def test_add_customer_behavior_features(aggregated_data):
    """Test the function that adds customer behavior features."""
    # Call the function to add customer behavior features
    result = add_customer_behavior_features(aggregated_data)

    # Define the expected result DataFrame with the 'customer_lifetime_value'
    # and 'transaction_growth_rate' columns added
    expected_result = pd.DataFrame({
        "customer_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "month": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "year": [2021] * 9,
        "total_transactions": [5, 3, 2, 4, 6, 1, 3, 2, 7],
        "customer_lifetime_value": [10, 10, 10, 11, 11, 11, 12, 12, 12],
        "transaction_growth_rate": [0, -0.4, -0.3333, 0, 0.5, -0.8333, 0, -0.3333, 2.5],
    })

    # Assert that the result matches the expected result
    pd.testing.assert_frame_equal(result, expected_result, rtol=1e-2)





def test_mean_encode_customer_id(aggregated_data):
    """Test the function that adds mean encoding for customer_id."""
    # Call the function to perform mean encoding
    result = mean_encode_customer_id(aggregated_data)

    # Define the expected result DataFrame with the mean encoded column
    expected_result = pd.DataFrame({
        "customer_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "month": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "year": [2021] * 9,
        "total_transactions": [5, 3, 2, 4, 6, 1, 3, 2, 7],
        "customer_mean_encoded_transactions": [3.33, 3.33, 3.33, 3.67, 3.67, 3.67, 4.0, 4.0, 4.0],
    })

    # Assert that the result matches the expected result
    pd.testing.assert_frame_equal(result, expected_result, rtol=1e-2)





def test_add_cyclical_month_features(aggregated_data):
    """Test the function that adds cyclical features for the month."""
    # Call the function to add cyclical month features
    result = add_cyclical_month_features(aggregated_data)

    # Define the expected result with cyclical month features added
    expected_result = pd.DataFrame({
        "customer_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "month": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "year": [2021] * 9,
        "total_transactions": [5, 3, 2, 4, 6, 1, 3, 2, 7],
        "month_sin": [0.5, 0.866, 1.0, 0.5, 0.866, 1.0, 0.5, 0.866, 1.0],
        "month_cos": [0.866, 0.5, 0.0, 0.866, 0.5, 0.0, 0.866, 0.5, 0.0],
    })

    # Assert that the result matches the expected result
    pd.testing.assert_frame_equal(result, expected_result, rtol=1e-3)





def test_add_numerical_year_feature(year_data):
    """Test the function that scales the year feature between 0 and 1."""
    # Call the function to scale the year feature
    result = add_numerical_year_feature(year_data)

    # Expected output DataFrame with the scaled 'year_scaled' column
    expected_result = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'year': [2018, 2019, 2020, 2021, 2022],
        'total_transactions': [5, 3, 7, 2, 6],
        'year_scaled': [0.0, 0.25, 0.5, 0.75, 1.0]  # Expected scaled values
    })

    # Assert that the result matches the expected result
    pd.testing.assert_frame_equal(result, expected_result)





def test_encode_features(data):
    """Test the encode_features function on the product_id column."""
    # Define the column to be one-hot encoded
    categorical_columns = ['product_id']

    # Call the encode_features function
    result = encode_features(data, categorical_columns)

    # Expected DataFrame after one-hot encoding
    expected_result = pd.DataFrame({
        'customer_id': [1, 1, 2, 2, 3],
        'date': [
            datetime(2021, 1, 1),
            datetime(2021, 1, 2),
            datetime(2021, 1, 3),
            datetime(2021, 1, 4),
            datetime(2021, 1, 5),
        ],
        'product_id_A': [1, 0, 1, 0, 1],
        'product_id_B': [0, 1, 0, 1, 0]
    })

    # Assert that the result matches the expected result
    pd.testing.assert_frame_equal(result, expected_result)





def test_create_smart_features(preprocessed_data):
    """Test the function that creates smart features."""
    # Call the function to create smart features
    result = create_smart_features(preprocessed_data)

    # Define the expected result based on the printed output
    expected_result = pd.DataFrame({
        "customer_id": [8752181] * 5 + [8752182] * 5,
        "month": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "year": [2017] * 10,
        "total_transactions": [1, 1, 0, 0, 1, 1, 0, 0, 1, 0],
        "transactions_lag_1": [0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
        "transactions_lag_2": [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
        "transactions_lag_3": [0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
        "moving_avg_3": [1.0, 1.0, 0.66, 0.33, 0.33, 1.0, 0.5, 0.33, 0.33, 0.33],
        "moving_avg_6": [1.0, 1.0, 0.66, 0.5, 0.6, 1.0, 0.5, 0.33, 0.5, 0.4],
        "recency": [0, 1, 0, 0, 3, 0, 0, 0, 3, 0],
        "frequency": [0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4],
        "customer_lifetime_value": [3, 3, 3, 3, 3, 2, 2, 2, 2, 2],
        "transaction_growth_rate": [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0],
        "customer_mean_encoded_transactions": [0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4],
        "month_sin": [0.5, 0.86, 1.0, 0.86, 0.5, 0.5, 0.86, 1.0, 0.86, 0.5],
        "month_cos": [0.86, 0.5, 0.0, -0.5, -0.86, 0.86, 0.5, 0.0, -0.5, -0.86],
        "year_scaled": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    })

    # Assert that the result matches the expected result
    pd.testing.assert_frame_equal(result, expected_result, rtol=1e-2)





def test_filter_data_by_range(preprocessed_data):
    """Test the function that creates filter data by range."""
    # Apply the filter_data_by_range function
    filtered_data = filter_data_by_range(preprocessed_data, (2017, 2), (2017, 4))

    # Define the expected output
    expected_data = pd.DataFrame({
        "customer_id": [8752181, 8752182],
        "product_id": ['G5HEPH9A2T', 'G5HEPH9A2T'],
        "date": [
            datetime(2017, 2, 15),
            datetime(2017, 4, 8),
        ],
        "year": [2017, 2017],
        "month": [2, 4],
    })

    # Assert that the filtered data is as expected
    pd.testing.assert_frame_equal(filtered_data, expected_data)





def test_split_data(feature_engineered_data):
    """Test the function that splits the data."""
    # Define training, validation, and test date ranges
    train_start, train_end = (2017, 1), (2017, 2)
    val_start, val_end = (2017, 3), (2017, 4)
    test_start, test_end = (2017, 5), (2017, 5)

    # Call the split_data function with the date ranges
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(
        feature_engineered_data,
        train_start, train_end,
        val_start, val_end,
        test_start, test_end
    )

    # Expected training data (dates between Jan and Feb 2017)
    expected_x_train = pd.DataFrame({
        "customer_id": [8752181, 8752181, 8752182],
        "product_id": ['G5HEPH9A2T', 'G5HEPH9A2T', 'G5HEPH9A2T'],
        "date": [
            datetime(2017, 1, 1),
            datetime(2017, 2, 15),
            datetime(2017, 1, 10),
        ],
        "year": [2017, 2017, 2017],
        "month": [1, 2, 1],
    })

    expected_y_train = pd.Series([5, 10, 4], name='total_transactions')

    # Expected validation data (dates between Mar and Apr 2017)
    expected_x_val = pd.DataFrame({
        "customer_id": [8752182],
        "product_id": ['G5HEPH9A2T'],
        "date": [
            datetime(2017, 4, 8),
        ],
        "year": [2017],
        "month": [4],
    })

    expected_y_val = pd.Series([6], name='total_transactions')

    # Expected test data (dates in May 2017)
    expected_x_test = pd.DataFrame({
        "customer_id": [8752181],
        "product_id": ['G5HEPH9A2T'],
        "date": [
            datetime(2017, 5, 23),
        ],
        "year": [2017],
        "month": [5],
    })

    expected_y_test = pd.Series([7], name='total_transactions')

    # Assert the splits
    pd.testing.assert_frame_equal(x_train, expected_x_train)
    pd.testing.assert_series_equal(y_train, expected_y_train)

    pd.testing.assert_frame_equal(x_val, expected_x_val)
    pd.testing.assert_series_equal(y_val, expected_y_val)

    pd.testing.assert_frame_equal(x_test, expected_x_test)
    pd.testing.assert_series_equal(y_test, expected_y_test)

# End of Python Script
