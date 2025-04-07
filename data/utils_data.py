import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time
from typing import Dict, Optional

def get_sp500_tickers() -> list:
    """
    Scrapes the current list of S&P 500 tickers from Wikipedia.

    :return: List of tickers
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        ticker = ticker.replace('.', '-')
        tickers.append(ticker)
    return tickers

def download_sp500_data(tickers: list, start_date: str, end_date: str, 
                         data_type: str = 'Close', chunk_size: int = 50) -> pd.DataFrame:
    """
    Downloads historical data for S&P 500 stocks.
    
    :param tickers: List of ticker symbols
    :param start_date: Start date in YYYY-MM-DD format
    :param end_date: End date in YYYY-MM-DD format
    :param data_type: Type of data to extract (Open, High, Low, Close, Volume, etc.)
    :param chunk_size: Number of tickers to process in each batch to avoid API limits
    :return: DataFrame with dates as index and tickers as columns
    """
    print(f"Downloading {data_type} data for {len(tickers)} stocks...")
    
    all_data = pd.DataFrame()
    
    # Process tickers in chunks to avoid potential API limitations
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        print(f"Processing tickers {i+1} to {min(i+chunk_size, len(tickers))}...")
        
        # Download data for this chunk of tickers
        data = yf.download(chunk, start=start_date, end=end_date, group_by='ticker')
        
        if len(chunk) == 1:
            # Handle the case of a single ticker (yfinance returns different format)
            ticker = chunk[0]
            single_df = pd.DataFrame(data[data_type])
            single_df.columns = [ticker]
            if all_data.empty:
                all_data = single_df
            else:
                all_data = pd.merge(all_data, single_df, left_index=True, right_index=True, how='outer')
        else:
            # Extract the specified data type from multi-ticker download
            for ticker in chunk:
                try:
                    ticker_data = pd.DataFrame(data[ticker][data_type])
                    ticker_data.columns = [ticker]
                    
                    if all_data.empty:
                        all_data = ticker_data
                    else:
                        all_data = pd.merge(all_data, ticker_data, left_index=True, right_index=True, how='outer')
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
        
        # Pause briefly to avoid hammering the API
        time.sleep(1)
    
    return all_data

def split_time_series_data(df: pd.DataFrame, train_size: float = 0.8, val_size: float = 0.1, 
                           test_size: float = 0.1, shuffle: bool = False) -> tuple:
    """
    Splits a time series DataFrame into training, validation, and test sets based on date.
    
    :param df: DataFrame with dates as index and tickers/features as columns
    :param train_size: Proportion of data to use for training (default: 0.8)
    :param val_size: Proportion of data to use for validation (default: 0.1)
    :param test_size: Proportion of data to use for testing (default: 0.1)
    :param shuffle: Whether to shuffle the data (default: False, as this is time series data)
    :return: Tuple of (train_df, val_df, test_df)
    """
    # Verify the split proportions sum to 1
    assert abs(train_size + val_size + test_size - 1.0) < 1e-10, "Split proportions must sum to 1"
    
    # Sort index to ensure data is in chronological order
    df = df.sort_index()
    
    # Calculate the split indices
    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    # Split the data
    if shuffle:
        # Note: Shuffling is generally not recommended for time series data
        indices = np.random.permutation(n)
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_df = df.iloc[train_indices]
        val_df = df.iloc[val_indices]
        test_df = df.iloc[test_indices]
    else:
        # Time-order preserving split
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
    
    print(f"Split sizes: Train: {len(train_df)} ({len(train_df)/n:.1%}), "
          f"Val: {len(val_df)} ({len(val_df)/n:.1%}), "
          f"Test: {len(test_df)} ({len(test_df)/n:.1%})")
    
    return train_df, val_df, test_df

def save_data_to_csv(data: pd.DataFrame, filename: str) -> None:
    """
    Saves the data to a CSV file.
    
    :param data: DataFrame to save
    :param filename: Name of the CSV file
    """
    data.to_csv(filename)
    print(f"Data saved to {filename}")

def load_data_from_csv(filename: str) -> pd.DataFrame:
    """
    Loads historical stock data from a CSV file.
    
    :param filename: Path to the CSV file
    :return: DataFrame with dates as index and tickers as columns
    """
    try:
        # Load the data, ensuring dates are parsed as datetime index
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"Successfully loaded data from {filename}")
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        return data
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return None

def split_time_series_data(df: pd.DataFrame, train_size: float = 0.8, val_size: float = 0.1, 
                           test_size: float = 0.1, shuffle: bool = False) -> tuple:
    """
    Splits a time series DataFrame into training, validation, and test sets based on date.
    
    :param df: DataFrame with dates as index and tickers/features as columns
    :param train_size: Proportion of data to use for training (default: 0.8)
    :param val_size: Proportion of data to use for validation (default: 0.1)
    :param test_size: Proportion of data to use for testing (default: 0.1)
    :param shuffle: Whether to shuffle the data (default: False, as this is time series data)
    :return: Tuple of (train_df, val_df, test_df)
    """
    # Verify the split proportions sum to 1
    assert abs(train_size + val_size + test_size - 1.0) < 1e-10, "Split proportions must sum to 1"
    
    # Sort index to ensure data is in chronological order
    df = df.sort_index()
    
    # Calculate the split indices
    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    # Split the data
    if shuffle:
        # Note: Shuffling is generally not recommended for time series data
        indices = np.random.permutation(n)
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_df = df.iloc[train_indices]
        val_df = df.iloc[val_indices]
        test_df = df.iloc[test_indices]
    else:
        # Time-order preserving split
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
    
    print(f"Split sizes: Train: {len(train_df)} ({len(train_df)/n:.1%}), "
          f"Val: {len(val_df)} ({len(val_df)/n:.1%}), "
          f"Test: {len(test_df)} ({len(test_df)/n:.1%})")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # File paths
    input_filename = 'data/sp500_csvs/sp500_historical_data.csv'
    train_filename = 'data/sp500_csvs/sp500_train.csv'
    val_filename = 'data/sp500_csvs/sp500_val.csv'
    test_filename = 'data/sp500_csvs/sp500_test.csv'
    
    # Option to download new data or use existing
    use_existing_data = False  # Set to False if you want to download fresh data
    
    if use_existing_data:
        # Load existing data
        data = load_data_from_csv(input_filename)
        if data is None:
            raise ValueError("Failed to load existing data. Please check the file path.")
    else:
        # Parameters for downloading new data
        start_date = '2000-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        data_type = 'Close'  # Options: Open, High, Low, Close, Adj Close, Volume
        
        # Get S&P 500 tickers
        print("Getting S&P 500 tickers...")
        tickers = get_sp500_tickers()
        print(f"Found {len(tickers)} tickers")
        
        # Download data
        data = download_sp500_data(tickers, start_date, end_date, data_type)
        
        # Save to CSV
        save_data_to_csv(data, input_filename)
    
    # Split data into train, validation, and test sets
    train_df, val_df, test_df = split_time_series_data(data)
    
    # Save the split datasets
    save_data_to_csv(train_df, train_filename)
    save_data_to_csv(val_df, val_filename)
    save_data_to_csv(test_df, test_filename)
    
    # Print summary statistics
    print("\nFull dataset:")
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Number of tickers with data: {len(data.columns)}")
    print(f"Percentage of missing values: {data.isna().mean().mean() * 100:.2f}%")
    
    print("\nTraining dataset:")
    print(f"Data shape: {train_df.shape}")
    print(f"Date range: {train_df.index.min()} to {train_df.index.max()}")
    
    print("\nValidation dataset:")
    print(f"Data shape: {val_df.shape}")
    print(f"Date range: {val_df.index.min()} to {val_df.index.max()}")
    
    print("\nTest dataset:")
    print(f"Data shape: {test_df.shape}")
    print(f"Date range: {test_df.index.min()} to {test_df.index.max()}")