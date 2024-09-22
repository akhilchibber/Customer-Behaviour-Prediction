"""This module hosts the feature engineering functions."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder # type: ignore
from akhil_take_home_assignment.data_preprocessing import aggregate_data_monthly

def add_lag_features(df: pd.DataFrame, lag_months: int) -> pd.DataFrame:
    """Add lag features to the aggregated data.

    Args:
        df (pd.DataFrame): Data with monthly transactions.
        lag_months (int): The number of lag features to create (e.g., 3 for 1 month ago, 2 months ago, etc.).

    Returns:
        pd.DataFrame: Dataframe with added lag features.
    """
    # Sort values by customer_id, year, and month to ensure correct lagging
    df = df.sort_values(by=['customer_id', 'year', 'month'])

    # Create lag features for each lag_month
    for lag in range(1, lag_months + 1):
        df[f'transactions_lag_{lag}'] = df.groupby('customer_id')['total_transactions'].shift(lag)

    # Fill NaN values that appear due to the shifting with 0
    df.fillna(0, inplace=True)
    for lag in range(1, lag_months + 1):
        df[f'transactions_lag_{lag}'] = df[f'transactions_lag_{lag}'].astype('int64')

    return df





def add_moving_average_features(df: pd.DataFrame, windows=[3, 6]) -> pd.DataFrame:
    """Add moving average features for total transactions over time for each customer.

    Args:
        df (pd.DataFrame): The aggregated data containing monthly transactions for each customer.
        windows (list): List of window sizes for calculating moving averages (default: [3, 6] months).

    Returns:
        pd.DataFrame: DataFrame with moving average features added.
    """
    # Ensure the data is sorted by customer, year, and month
    df = df.sort_values(by=['customer_id', 'year', 'month'])

    # Create moving averages for each window size
    for window in windows:
        df[f'moving_avg_{window}'] = df.groupby('customer_id')['total_transactions'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

    return df





def calculate_recency(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the recency in months for each customer.

    This is the number of months since their last actual transaction.

    Args:
        df (pd.DataFrame): Dataframe containing customer transactions with 'year',
                           'month', and 'total_transactions' columns.

    Returns:
        pd.DataFrame: Updated dataframe with a 'recency' column indicating the
                      number of months since the last actual transaction.
    """
    # Ensure the data is sorted by customer_id, year, and month
    df = df.sort_values(by=['customer_id', 'year', 'month'])

    # Combine year and month into a 'year_month' column for easier date comparison
    df['year_month'] = df['year'] * 12 + df['month']  # Convert year and month into a numeric scale (year * 12 + month)

    # Initialize last_transaction to NaN initially for all customers
    df['last_transaction_year_month'] = np.nan

    # Iterate over each customer to calculate recency based on non-zero transactions
    for customer_id, customer_data in df.groupby('customer_id'):
        # Track the last time a transaction happened
        last_transaction_time = np.nan

        # Iterate through the customer's data in chronological order
        for idx, row in customer_data.iterrows():
            # Check if the customer made any transaction in the current month
            if row['total_transactions'] > 0:
                # If this is a valid transaction month, calculate the recency based on the last transaction
                if not pd.isna(last_transaction_time):
                    df.at[idx, 'last_transaction_year_month'] = last_transaction_time

                # Update the last transaction time to the current year-month
                last_transaction_time = row['year_month']

    # Calculate recency by subtracting the last transaction month from the current month
    df['recency'] = df['year_month'] - df['last_transaction_year_month']

    # Fill NaN values in recency (first transactions) with a large number (e.g., 9999)
    df['recency'] = df['recency'].fillna(0)

    # Explicitly cast 'recency' column to integer
    df['recency'] = df['recency'].astype(int)

    # Drop temporary columns
    df.drop(columns=['year_month', 'last_transaction_year_month'], inplace=True)

    return df





def add_recency_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add recency and frequency features to the data for each customer.

    Args:
        df (pd.DataFrame): The aggregated data containing monthly transactions for each customer.

    Returns:
        pd.DataFrame: DataFrame with recency and frequency features added.
    """
    # Calculate Recency: Add recency column based on actual transaction months
    df = calculate_recency(df)

    # Calculate Frequency: Average number of transactions per month for each customer
    df['frequency'] = df.groupby('customer_id')['total_transactions'].transform('mean')

    return df





def add_customer_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add customer behavior features: lifetime value and transaction growth rate.

    Args:
        df (pd.DataFrame): The aggregated data containing monthly transactions for each customer.

    Returns:
        pd.DataFrame: DataFrame with customer behavior features added.
    """
    # Calculate Customer Lifetime Value: Total transactions for each customer over their entire history
    df['customer_lifetime_value'] = df.groupby('customer_id')['total_transactions'].transform('sum')

    # Calculate Transaction Growth Rate: Percent change in transactions compared to the previous month
    df['transaction_growth_rate'] = df.groupby('customer_id')['total_transactions'].pct_change().fillna(0)

    # Replace infinite values with np.nan or capped values
    df['transaction_growth_rate'] = df['transaction_growth_rate'].replace([np.inf, -np.inf], np.nan)

    # Optionally: Fill NaN values with 0, or you can use more advanced imputation strategies later
    df['transaction_growth_rate'] = df['transaction_growth_rate'].fillna(0)

    return df





def mean_encode_customer_id(data: pd.DataFrame) -> pd.DataFrame:
    """Perform mean encoding for the 'customer_id' column.

    This is done by calculating the average number of transactions
    per customer across months.

    Args:
        data (pd.DataFrame): Preprocessed data containing customer transactions

    Returns:
        pd.DataFrame: Dataframe with mean encoding for customer_id
    """
    # Calculate the mean number of transactions for each customer
    customer_mean = data.groupby('customer_id')['total_transactions'].mean().reset_index()
    customer_mean.columns = pd.Index(['customer_id', 'customer_mean_encoded_transactions'])

    # Merge the mean encoded feature back into the original data
    data = pd.merge(data, customer_mean, on='customer_id', how='left')

    return data





def add_cyclical_month_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical representation for the 'month' column using sine and cosine transformations.

    Args:
        df (pd.DataFrame): The aggregated data containing monthly transactions for each customer.

    Returns:
        pd.DataFrame: DataFrame with cyclical month features added.
    """
    # Create sine and cosine transformations for the month column
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Round values to a precision that matches the expected results
    df['month_sin'] = df['month_sin'].round(3)
    df['month_cos'] = df['month_cos'].round(3)

    return df





def add_numerical_year_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the 'year' feature into a scaled numerical format for machine learning models.

    Args:
        df (pd.DataFrame): The data containing the 'year' feature.

    Returns:
        pd.DataFrame: DataFrame with a scaled 'year' feature.
    """
    # Scale the 'year' feature between 0 and 1 or between a specific range (min/max normalization)
    min_year = df['year'].min()
    max_year = df['year'].max()

    if min_year == max_year:
        # If all years are the same, set year_scaled to 0
        df['year_scaled'] = 0
    else:
        # Normalizing year between 0 and 1
        df['year_scaled'] = (df['year'] - min_year) / (max_year - min_year)

    return df





# This function has not been used anywhere. I still thought to prepare it
# because it was part of the given assignment. Maybe we can use it in future.
def encode_features(df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    """One-hot encode categorical features.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        categorical_columns (list): List of categorical columns to be one-hot encoded.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded features.
    """
    # Initialize OneHotEncoder from scikit-learn
    encoder = OneHotEncoder(sparse_output=False)

    # Fit and transform the categorical columns
    encoded = encoder.fit_transform(df[categorical_columns])

    # Create a DataFrame with encoded values
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_columns),
        index=df.index
    )

    # Convert the encoded columns to integer type
    encoded_df = encoded_df.astype(int)

    # Drop the original categorical columns and join the one-hot encoded columns
    df = df.drop(columns=categorical_columns).join(encoded_df)

    return df





def create_smart_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create smart features by applying various feature engineering techniques.

    These techniques include aggregation, lag features, moving averages, recency
    and frequency features, customer behavior features, mean encoding, and cyclical
    month features.

    Args:
        data (pd.DataFrame): DataFrame containing the original data with customer transaction details.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    # Step 1: Aggregate data into monthly summaries
    aggregated_data = aggregate_data_monthly(data)

    # Step 2: Add lag features (e.g., create 3 months of lag features)
    lag_months = 3  # Example for 3-month lag
    data_with_lag = add_lag_features(aggregated_data, lag_months=lag_months)

    # Step 3: Add moving average features (e.g., 3-month and 6-month moving averages)
    moving_average_windows = [3, 6]  # Example windows
    data_with_moving_avg = add_moving_average_features(data_with_lag, windows=moving_average_windows)

    # Step 4: Add recency and frequency features
    data_with_recency_frequency = add_recency_frequency_features(data_with_moving_avg)

    # Step 5: Add customer behavior features (lifetime value and transaction growth rate)
    data_with_customer_behavior = add_customer_behavior_features(data_with_recency_frequency)

    # Step 6: Add mean encoding for customer_id
    data_with_mean_encoding = mean_encode_customer_id(data_with_customer_behavior)

    # Step 7: Add cyclical month features (sine and cosine transformations)
    data_with_cyclical_month = add_cyclical_month_features(data_with_mean_encoding)

    # Step 8: Add numerical year feature (scale year between 0 and 1)
    final_data = add_numerical_year_feature(data_with_cyclical_month)

    return final_data





def filter_data_by_range(data: pd.DataFrame, start: tuple, end: tuple) -> pd.DataFrame:
    """Filter data based on the start and end dates provided.

    Args:
        data (pd.DataFrame): The preprocessed data containing 'year' and 'month'.
        start (tuple): Start date as (year, month).
        end (tuple): End date as (year, month).

    Returns:
        pd.DataFrame: Filtered data within the specified date range.
    """
    start_year, start_month = start
    end_year, end_month = end

    filtered_data =  data[((data['year'] > start_year) |
                           ((data['year'] == start_year) & (data['month'] >= start_month))) &
                          ((data['year'] < end_year) |
                           ((data['year'] == end_year) & (data['month'] <= end_month)))]

    # Reset the index here
    return filtered_data.reset_index(drop=True)





def split_data(data: pd.DataFrame, train_start: tuple, train_end: tuple,
               val_start: tuple, val_end: tuple, test_start: tuple, test_end: tuple):
    """Split the data into training, validation, and test sets based on the provided date ranges.

    Args:
        data (pd.DataFrame): The preprocessed and feature-engineered data.
        train_start (tuple): Start of the training period (year, month).
        train_end (tuple): End of the training period (year, month).
        val_start (tuple): Start of the validation period (year, month).
        val_end (tuple): End of the validation period (year, month).
        test_start (tuple): Start of the test period (year, month).
        test_end (tuple): End of the test period (year, month).

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test: Data split into features and targets for each set.
    """
    # Filter data for the training, validation, and test periods
    train_data = filter_data_by_range(data, train_start, train_end)
    val_data = filter_data_by_range(data, val_start, val_end)
    test_data = filter_data_by_range(data, test_start, test_end)

    # Split the training data into features (X) and target (y)
    x_train = train_data.drop(columns=['total_transactions']).reset_index(drop=True)
    y_train = train_data['total_transactions'].reset_index(drop=True)

    # Split the validation data into features (X) and target (y)
    x_val = val_data.drop(columns=['total_transactions']).reset_index(drop=True)
    y_val = val_data['total_transactions'].reset_index(drop=True)

    # Split the test data into features (X) and target (y)
    x_test = test_data.drop(columns=['total_transactions']).reset_index(drop=True)
    y_test = test_data['total_transactions'].reset_index(drop=True)

    return x_train, y_train, x_val, y_val, x_test, y_test

# End of Python Script
