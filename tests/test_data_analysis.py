"""This module includes Unit tests for data exploration functions!"""

from akhil_take_home_assignment.data_analysis import total_number_transactions, \
    transactions_frequency, top_products, add_date_columns, calculate_yearly_transactions,\
    calculate_month_year_transactions, calculate_seasonal_months, calculate_date_range

import pandas as pd
from datetime import datetime

def test_add_date_columns(data):
    """Test the addition of year, month, and month_name columns."""
    result = add_date_columns(data)

    expected_result = pd.DataFrame(
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
            "year": [2021, 2021, 2021, 2021, 2021],
            "month": [1, 1, 1, 1, 1],
            "month_name": ["January", "January", "January", "January", "January"],
        }
    )

    # Use pd.testing.assert_frame_equal to avoid false negatives due to index differences
    pd.testing.assert_frame_equal(result, expected_result)





def test_calculate_yearly_transactions(data):
    """Test the calculation of total transactions by year."""
    # Call the function to get the result
    result = calculate_yearly_transactions(data)

    # Expected result DataFrame
    expected_result = pd.DataFrame({
        "year": [2021],
        "total_transactions": [5]
    })

    # Assert that the result matches the expected result
    pd.testing.assert_frame_equal(result, expected_result)





def test_calculate_month_year_transactions(data):
    """Test the total transactions grouped by year and month."""
    result = calculate_month_year_transactions(data)

    # Define the expected result with the correct total transactions
    expected_result = pd.DataFrame(
        {
            'year': [2021],
            'month': [1],
            'month_name': ['January'],
            'total_transactions': [5],  # All 5 transactions are aggregated into one row
        }
    )

    # Use pd.testing.assert_frame_equal to compare the DataFrames
    pd.testing.assert_frame_equal(result, expected_result)





def test_calculate_seasonal_months(seasonal_data):
    """Test identifying months with above-average transactions."""
    result = calculate_seasonal_months(seasonal_data)

    # Define the expected result based on the fixture data
    expected_months = [2, 4, 6, 7, 10, 12]

    # Check if the function's result matches the expected result
    assert list(map(int, result)) == expected_months






def test_total_number_transactions(data):
    """Test the total number of transactions per customer."""
    result = total_number_transactions(data)
    assert result.equals(
        pd.DataFrame(
            {
                "customer_id": [1, 2, 3],
                "total_transactions": [2, 2, 1],
            }
        )
    )





def test_transactions_frequency(data):
    """Test the transaction frequency per month for a product id in a year."""
    result = transactions_frequency(data, "A", "2021")
    assert result.equals(
        pd.DataFrame(
            {
                "month": [1],
                "frequency": [3],
            }
        )
    )





def test_calculate_date_range(monthly_transaction_data):
    """Test the calculation of the six-month date range."""
    # Call the calculate_date_range function using the dataset
    start_time, end_time = calculate_date_range(monthly_transaction_data)

    # Expected result as a DataFrame for easy comparison
    expected_result = pd.DataFrame({
        'start_time': [datetime(2020, 11, 9)],  # Example based on fixture data
        'end_time': [datetime(2021, 5, 9)]
    })

    # Actual result in a DataFrame
    result = pd.DataFrame({
        'start_time': [start_time],
        'end_time': [end_time]
    })

    # Assert that the result matches the expected result
    pd.testing.assert_frame_equal(result, expected_result)





def test_top_products(transaction_data):
    """Test the top 5 products over the last six month."""
    result = top_products(transaction_data)

    print(result)

    assert result.equals(
        pd.DataFrame(
            {
               "product_id": ["A", "B", "C", "D", "E"],
               "total_transactions": [4, 3, 2, 1, 1],
            }
        )
    )

# End of Python Script
