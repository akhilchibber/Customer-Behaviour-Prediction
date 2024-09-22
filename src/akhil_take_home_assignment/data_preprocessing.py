"""This module provides data preprocessing functions!"""

import pandas as pd
from pathlib import Path

def load_data(path: Path) -> pd.DataFrame:
    """Load data from a directory with multiple CSV files or a single CSV file.

    Args:
        path (Path): Path to a CSV file or a directory containing CSV files.

    Returns:
        pd.DataFrame: Data loaded from the CSV file(s).
    """
    path = Path(path)

    if path.is_file():
        # If the path is a file, read the single CSV file
        return pd.read_csv(path)
    elif path.is_dir():
        # If the path is a directory, list all CSV files in the directory
        csv_files = list(path.glob("*.csv"))

        # Read each CSV file and concatenate into a single DataFrame
        dataframes = [pd.read_csv(csv_file) for csv_file in csv_files]

        # Concatenate all DataFrames
        combined_df = pd.concat(dataframes, ignore_index=True)

        return combined_df
    else:
        raise ValueError("The provided path is neither a file nor a directory.")





def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the data and return a cleaned DataFrame.

    Criteria:
    - Remove duplicates
    - Remove rows with missing values

    Args:
        data (pd.DataFrame): Data to be cleaned

    Returns:
        pd.DataFrame: Cleaned data
    """
    # Remove duplicates
    data_cleaned = data.drop_duplicates()

    # Remove rows with missing values
    data_cleaned = data_cleaned.dropna()

    # Keep only the required columns: 'customer_id', 'product_id', and 'date'
    data_cleaned = data_cleaned[['customer_id', 'product_id', 'date']]

    return data_cleaned





def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data and return a preprocessed DataFrame.

    Criteria:
    - Convert date column to datetime
    - Sort the data by date

    Args:
        data (pd.DataFrame): Data to be preprocessed

    Returns:
        pd.DataFrame: Preprocessed data
    """
    # Convert 'date' column to datetime
    data['date'] = pd.to_datetime(data['date'])

    # Sort the data by 'date'
    data = data.sort_values(by='date').reset_index(drop=True)

    return data





def aggregate_data_monthly(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate data into monthly summaries.

    Args:
        data (pd.DataFrame): Preprocessed transaction data.

    Returns:
        pd.DataFrame: Aggregated monthly data with customer_id, month, year, and total_transactions.
    """
    # Extract year and month from the date column
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month

    # Create a dataframe with all possible combinations of customer_id, year, and month
    all_customers = data['customer_id'].unique()
    date_range = pd.date_range(
        data['date'].min(),
        data['date'].max() + pd.DateOffset(months=1), freq='ME')
    all_months_years = pd.DataFrame(
        [(customer, date.month, date.year) for customer in all_customers for date in date_range],
        columns=['customer_id', 'month', 'year'])

    # Aggregate transactions by customer, month, and year
    monthly_agg = data.groupby(['customer_id', 'year', 'month']).size().reset_index()

    # Rename the column after reset_index
    monthly_agg.columns = pd.Index(['customer_id', 'year', 'month', 'total_transactions'])

    # Merge to ensure all combinations of customer_id and months are included
    aggregated_data = pd.merge(all_months_years, monthly_agg, on=['customer_id', 'year', 'month'], how='left')

    # Fill missing transactions with 0
    # aggregated_data['total_transactions'].fillna(0, inplace=True)
    aggregated_data['total_transactions'] = aggregated_data['total_transactions'].fillna(0).astype(int)
    # aggregated_data['total_transactions'] = aggregated_data['total_transactions'].astype(int)

    return aggregated_data

# End of Python Script
