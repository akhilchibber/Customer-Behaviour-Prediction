"""This module includes Unit tests for Data Pre-Processing functions!"""

from akhil_take_home_assignment.data_preprocessing import load_data, \
    clean_data, preprocess_data, aggregate_data_monthly
from datetime import datetime
import pandas as pd
from pathlib import Path

def test_load_data(data):
    """Test the load data function."""
    # Use the fixture-provided data
    data.to_csv("test_data_3.csv", index=False)

    # Test if load_data can read it
    transactions = load_data(Path("test_data_3.csv"))

    # Assert that the loaded file is a DataFrame
    assert isinstance(transactions, pd.DataFrame)





def test_clean_data(unclean_data):
    """Test the clean data function."""
    cleaned_data = clean_data(unclean_data)

    print(cleaned_data)

    expected_cleaned_data = pd.DataFrame(
        {
            "customer_id": ["XYZ", "XYZ", "XYZ", "ABC", "ABC", "DEF"],
            "product_id": ["A", "A", "B", "A", "B", "A"],
            "date": [
                datetime(2021, 1, 1),
                datetime(2021, 1, 1),
                datetime(2021, 1, 2),
                datetime(2021, 1, 3),
                datetime(2021, 1, 4),
                datetime(2021, 1, 5),
            ],
        }
    )

    assert cleaned_data.equals(expected_cleaned_data)





def test_preprocess_data(unprocessed_data):
    """Test the preprocess data function."""
    preprocessed_data = preprocess_data(unprocessed_data)

    print(preprocessed_data)

    # Expected preprocessed DataFrame
    expected_preprocessed_data = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3],
            "product_id": ["A", "B", "A", "B", "A"],
            "date": [
                pd.Timestamp("2021-01-01 07:20:15.000000+00:00"),
                pd.Timestamp("2021-01-02 08:45:30.000000+00:00"),
                pd.Timestamp("2021-01-03 13:30:00.000000+00:00"),
                pd.Timestamp("2021-01-04 09:05:10.000000+00:00"),
                pd.Timestamp("2021-01-05 10:15:30.000000+00:00"),
            ],
        }
    )

    # Assert that the actual preprocessed data equals the expected data
    pd.testing.assert_frame_equal(preprocessed_data, expected_preprocessed_data)

    # Assert the dtype of the 'date' column is datetime64[ns, UTC]
    assert preprocessed_data['date'].dtype == 'datetime64[ns, UTC]'

# def test_preprocess_data(data):
#     """Test the preprocess data function."""
#     preprocessed_data = preprocess_data(data)
#     assert preprocessed_data.equals(
#         pd.DataFrame(
#             {
#                 "customer_id": [1, 1, 2, 2, 3],
#                 "product_id": ["A", "B", "A", "B", "A"],
#                 "date": [
#                     datetime(2021, 1, 1),
#                     datetime(2021, 1, 2),
#                     datetime(2021, 1, 3),
#                     datetime(2021, 1, 4),
#                     datetime(2021, 1, 5),
#                 ],
#             }
#         )
#     )





def test_aggregate_data_monthly(preprocessed_data):
    """Test the function that aggregates data into monthly summaries."""
    # Call the function to aggregate the data
    result = aggregate_data_monthly(preprocessed_data)

    # Print the original fixture data
    print("Fixture Data:")
    print(preprocessed_data)

    # Print the output of the function
    print("\nAggregated Result:")
    print(result)

    # Define the expected result with the actual months for customer_id 8752181 and 8752182 included
    expected_result = pd.DataFrame({
        "customer_id": [8752181] * 5 + [8752182] * 5,  # Cover only Jan-May for both customers
        "month": [1, 2, 3, 4, 5] * 2,  # Months from Jan (1) to May (5)
        "year": [2017] * 10,  # All transactions are in 2017
        "total_transactions": [
            1, 1, 0, 0, 1,  # Customer 8752181's transactions in Jan, Feb, May
            1, 0, 0, 1, 0   # Customer 8752182's transactions in Jan, Apr
        ]
    })

    # Assert that the result matches the expected result
    pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)

# End of Python Script
