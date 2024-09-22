"""This module hosts the business logic of the application."""

import pandas as pd
from dateutil.relativedelta import relativedelta

def add_date_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Add columns for year, month, and month name.

    Args:
        data (pd.DataFrame): Data containing a 'date' column.

    Returns:
        pd.DataFrame: Data with added 'year', 'month', and 'month_name' columns.
    """
    data['year'] = data['date'].dt.year.astype('int64')
    data['month'] = data['date'].dt.month.astype('int64')
    data['month_name'] = data['date'].dt.strftime('%B')
    return data





def calculate_yearly_transactions(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate total transactions by year.

    Args:
        data (pd.DataFrame): Data containing customer transactions.

    Returns:
        pd.DataFrame: Total transactions grouped by year.
    """
    # Ensure that 'date' column is in datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Extract the 'year' column
    data['year'] = data['date'].dt.year.astype('int64')

    # yearly_transactions = data.groupby('year').size().reset_index(name='total_transactions')
    yearly_transactions = data.groupby('year').agg(
        total_transactions=('year', 'size')).reset_index()

    print("\nTotal Transactions by Year:")
    print(yearly_transactions)
    return yearly_transactions





def calculate_month_year_transactions(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate total transactions by year and month.

    Args:
        data (pd.DataFrame): Data containing customer transactions.

    Returns:
        pd.DataFrame: Total transactions grouped by year and month.
    """
    # Use the existing function to add year, month, and month_name columns
    data = add_date_columns(data)

    # month_year_transactions = data.groupby(
    #     ['year', 'month', 'month_name']).size().reset_index(name='total_transactions')
    month_year_transactions = data.groupby(['year', 'month', 'month_name']).agg(
        total_transactions=('year', 'size')).reset_index()

    print("\nTotal Transactions by Month-Year Combination:")
    print(month_year_transactions)
    return month_year_transactions





def calculate_seasonal_months(data: pd.DataFrame) -> list:
    """Identify months with above-average total transactions across all years.

    Args:
        data (pd.DataFrame): DataFrame containing total transactions with 'month' and 'total_transactions' columns.

    Returns:
        list: A list of months where the total transactions are above average.
    """
    monthly_avg = data.groupby('month')['total_transactions'].mean().reset_index()
    seasonal_months = monthly_avg[monthly_avg['total_transactions'] >
                                  monthly_avg['total_transactions'].mean()]['month'].tolist()
    return seasonal_months





def total_number_transactions(data: pd.DataFrame) -> pd.DataFrame:
    """Return the total number of transactions per customer.

    Args:
        data (pd.DataFrame): Data containing customer transactions

    Returns:
        pd.DataFrame: Total number of transactions per customer, sorted in descending order
    """
    # Group by customer_id and count the transactions
    # transaction_counts = data.groupby('customer_id').size().reset_index(name='total_transactions')
    transaction_counts = data.groupby('customer_id').agg(
        total_transactions=('customer_id', 'size')).reset_index()

    # Sort in descending order by the number of transactions
    transaction_counts = transaction_counts.sort_values(by='total_transactions', ascending=False)

    return transaction_counts





def transactions_frequency(data: pd.DataFrame, product_id: str, year: str) -> pd.DataFrame:
    """Return transaction frequency per month for a product id in a year.

    Args:
        data (pd.DataFrame): Data containing transactions
        product_id (str): Product ID
        year (str): Year (in 'YYYY' format)

    Returns:
        pd.DataFrame: Transaction frequency per month for a product id in a year
    """
    # Convert 'date' column to datetime if not already in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])

    # Filter data for the specified product_id and year
    filtered_data = data[
        (data['product_id'] == product_id) & (data['date'].dt.year == int(year))
    ]

    # Group by month and count transactions
    # monthly_frequency = filtered_data.groupby(
    #     filtered_data['date'].dt.month).size().reset_index(name='frequency')
    monthly_frequency = filtered_data.groupby(filtered_data['date'].dt.month).agg(
        frequency=('date', 'size')).reset_index()


    # Rename the first column to 'month' (it's actually the month index)
    monthly_frequency.columns = pd.Index(['month', 'frequency'])

    # Ensure the types of the columns are explicitly set to match the expected DataFrame in the test
    monthly_frequency['month'] = monthly_frequency['month'].astype(int)
    monthly_frequency['frequency'] = monthly_frequency['frequency'].astype(int)

    return monthly_frequency





def calculate_date_range(data: pd.DataFrame) -> tuple:
    """Calculate the date range for the last six months based on the most recent date in the dataset.

    Args:
        data (pd.DataFrame): A DataFrame containing a 'date' column with datetime objects.

    Returns:
        tuple: A tuple containing two dates:
            - The start date: exactly six months prior to the latest date in the 'date' column.
            - The end date: the latest date found in the 'date' column.
    """
    # Find the most recent date in the dataset
    latest_date = data['date'].max()

    # Ensure the date is timezone-naive, if it's a datetime object
    if pd.api.types.is_datetime64_any_dtype(latest_date):
        latest_date = latest_date.tz_localize(None)

    # Calculate the date range: six months back from the latest date
    end_time = latest_date
    start_time = end_time - relativedelta(months=6)

    return start_time, end_time





def top_products(data: pd.DataFrame) -> pd.DataFrame:
    """Return the top 5 products over the last six months.

    Args:
        data (pd.DataFrame): Data containing transactions
        start_time (datetime): Start time to filter the data

    Returns:
        pd.DataFrame: Top 5 products over the last six months by sales
    """
    # Convert 'date' column to datetime if not already in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])

    # Ensure the 'date' column is timezone-naive
    data['date'] = data['date'].dt.tz_localize(None)

    # Use the calculate_date_range function to get the date range
    six_months_ago, end_time = calculate_date_range(data)

    # Filter data for the last six months
    filtered_data = data[
        (data['date'] >= six_months_ago) & (data['date'] <= end_time)
    ]

    # Group by product_id and count transactions, then sort to find the top 5 products
    # top_products = filtered_data.groupby('product_id').size().reset_index(name='total_transactions')
    top_products = filtered_data.groupby('product_id').agg(
        total_transactions=('product_id', 'size')).reset_index()


    # Sort by total_transactions in descending order and get the top 5
    top_products = top_products.sort_values(by='total_transactions', ascending=False).head(5)

    # Remove the second sort by product_id to preserve the order by total_transactions
    top_products = top_products.reset_index(drop=True)  # Keep the index clean

    print(top_products)

    return top_products

# End of Python Script
