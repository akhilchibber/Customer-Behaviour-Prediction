"""All functional tests for the customer behavior module.

We are using a single file given the size of this project. However,
feel free to start splitting the tests, if you feel it is necessary.

"""

import os
import shutil
import pytest
import pandas as pd
from datetime import datetime
from hypothesis import given, strategies as st

from akhil_take_home_assignment.data_preprocessing import load_data, clean_data
from akhil_take_home_assignment.data_analysis import total_number_transactions, \
    transactions_frequency, top_products

# This was already part of the provided script, but I don't feel this has any use
# pytestmark = pytest.mark.functional

# Define the shared strategy once
sample_strategy = st.lists(
    st.lists(
        st.tuples(
            st.text(
                min_size=10,
                max_size=10,
                alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
            ),
            st.integers(min_value=1000000, max_value=9999999),
            st.datetimes(
                min_value=datetime(2018, 1, 1), max_value=datetime(2022, 1, 1)
            ),
        ),
        min_size=2,
        max_size=10,
    ),
    min_size=1,
    max_size=10,
)

@pytest.mark.dependency(name="s1")
@given(samples=sample_strategy)
def test_load_data(samples):
    """Tests can load mulipart transactions data into a single DataFrame."""
    shutil.rmtree("/tmp/data/", ignore_errors=True)
    dataset_path = "/tmp/data/"
    os.makedirs(dataset_path, exist_ok=True)

    length = 0
    for idx, sample in enumerate(samples):
        with open(os.path.join(dataset_path, f"transaction_{idx}.csv"), "w") as f:
            pd.DataFrame(sample, columns=["customer_id", "product_id", "date"]).to_csv(  # type: ignore
                f
            )
        length += len(sample)

    transactions = load_data(dataset_path)
    assert isinstance(transactions, pd.DataFrame)
    assert len(transactions) == length





@pytest.mark.dependency(name="s2", depends=["s1"])
@given(samples=sample_strategy)
def test_total_number_transactions(samples):
    """Test the total number of transactions per customer using hypothesis."""
    # Convert samples to DataFrame
    df = pd.DataFrame(
        [
            (customer_id, product_id, date)
            for sample in samples
            for customer_id, product_id, date in sample
        ],
        columns=["customer_id", "product_id", "date"]
    )

    # Ensure the DataFrame has at least some rows to test
    if df.empty:
        return  # Skip if no data

    # Call the total_number_transactions function
    result = total_number_transactions(df)

    # Ensure that customer_id and total_transactions columns exist
    assert "customer_id" in result.columns
    assert "total_transactions" in result.columns

    # Check that the number of unique customer_id entries in result matches the input
    assert result["customer_id"].nunique() == df["customer_id"].nunique()

    # Check that total_transactions are positive or zero
    assert result["total_transactions"].ge(0).all()

    # Ensure no duplicate customer_ids in the result
    assert result["customer_id"].is_unique





@pytest.mark.dependency(name="s3", depends=["s2"])
@given(samples=sample_strategy)
def test_transactions_frequency(samples):
    """Test the transaction frequency per month for a product id in a year using hypothesis."""
    # Convert samples to DataFrame
    df = pd.DataFrame(
        [
            (customer_id, product_id, date)
            for sample in samples
            for customer_id, product_id, date in sample
        ],
        columns=["customer_id", "product_id", "date"]
    )

    # Ensure the DataFrame has at least some rows to test
    if df.empty:
        return  # Skip if no data

    # Select a random product_id from the data and a random year for testing
    product_id = df['product_id'].iloc[0] if len(df) > 0 else None
    year = df['date'].dt.year.min() if len(df) > 0 else None

    if product_id is None or year is None:
        return  # Skip if no valid product_id or year found

    # Call the transactions_frequency function
    result = transactions_frequency(df, product_id, str(year))

    # Ensure that 'month' and 'frequency' columns exist in the result
    assert "month" in result.columns
    assert "frequency" in result.columns

    # Ensure that the month column contains valid month values between 1 and 12
    assert result["month"].between(1, 12).all()

    # Ensure that the frequency column contains values greater than or equal to 0
    assert result["frequency"].ge(0).all()





@pytest.mark.dependency(name="s4", depends=["s3"])
@given(samples=sample_strategy)
def test_top_products(samples):
    """Test the top 5 products over the last six months using hypothesis data."""
    # Convert samples to DataFrame
    df = pd.DataFrame(
        [
            (customer_id, product_id, date)
            for sample in samples
            for customer_id, product_id, date in sample
        ],
        columns=["customer_id", "product_id", "date"]
    )

    print("\nGenerated DataFrame from Hypothesis samples:")
    print(df.head())

    # Ensure the DataFrame has at least some rows to test
    if df.empty:
        print("Empty DataFrame, skipping the test.")
        return  # Skip if no data

    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    print("\nDataFrame after date conversion:")
    print(df.head())

    # Call the top_products function
    result = top_products(df)

    print("\nResult from top_products function:")
    print(result)

    # Assertions
    # Ensure that 'product_id' and 'total_transactions' columns exist in the result
    assert "product_id" in result.columns, "Missing 'product_id' column"
    assert "total_transactions" in result.columns, "Missing 'total_transactions' column"

    # Ensure that total_transactions are positive
    assert result["total_transactions"].ge(0).all(), "Some 'total_transactions' are negative"

    # Ensure that there are at most 5 products in the result
    assert len(result) <= 5, f"More than 5 products found: {len(result)}"

    # Ensure no duplicate product_ids in the result
    assert result["product_id"].is_unique, "Duplicate product_ids found in the result"





@pytest.mark.dependency(name="s5", depends=["s4"])
@given(samples=sample_strategy)
def test_clean_data(samples):
    """Test the clean_data function using Hypothesis-generated data."""
    # Convert samples to DataFrame
    df = pd.DataFrame(
        [
            (customer_id, product_id, date)
            for sample in samples
            for customer_id, product_id, date in sample
        ],
        columns=["customer_id", "product_id", "date"]
    )

    # Call the clean_data function
    cleaned_data = clean_data(df)

    # Ensure the cleaned data has no duplicates
    assert cleaned_data.duplicated().sum() == 0, "There are duplicates in the cleaned data"

    # Ensure the cleaned data has no missing values
    assert cleaned_data.isna().sum().sum() == 0, "There are missing values in the cleaned data"

    # Ensure the cleaned data only has the 'customer_id', 'product_id', and 'date' columns
    assert set(cleaned_data.columns) == {"customer_id", "product_id", "date"}, "Unexpected columns in the cleaned data"

    # Ensure the data types are correct
    assert pd.api.types.is_string_dtype(cleaned_data['customer_id']), "customer_id should be a string"
    assert pd.api.types.is_integer_dtype(cleaned_data['product_id']), "product_id should be an integer"
    assert pd.api.types.is_datetime64_any_dtype(cleaned_data['date']), "date should be a datetime"

# End of Python Script
