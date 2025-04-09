import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time
import os

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
    
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        print(f"Processing tickers {i+1} to {min(i+chunk_size, len(tickers))}...")
        
        data = yf.download(chunk, start=start_date, end=end_date, group_by='ticker')
        
        if len(chunk) == 1:
            ticker = chunk[0]
            single_df = pd.DataFrame(data[data_type])
            single_df.columns = [ticker]
            if all_data.empty:
                all_data = single_df
            else:
                all_data = pd.merge(all_data, single_df, left_index=True, right_index=True, how='outer')
        else:
            for ticker in chunk:
                try:
                    ticker_data = pd.DataFrame(data[ticker][data_type])
                    ticker_data.columns = [ticker]
                    
                    if all_data.empty:
                        all_data = ticker_data
                    else:
                        all_data = pd.merge(all_data, ticker_data, left_index=True, right_index=True, how='outer')
                except Exception as e:
                    print(f"Error processing {ticker}: {e}, skipping...")
        
        # avoid hammering the API
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
    assert abs(train_size + val_size + test_size - 1.0) < 1e-10, "Split proportions must sum to 1"
    
    df = df.sort_index()
    
    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    if shuffle:
        indices = np.random.permutation(n)
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_df = df.iloc[train_indices]
        val_df = df.iloc[val_indices]
        test_df = df.iloc[test_indices]
    else: # time-order preserving split
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
    
    print(f"Split sizes: Train: {len(train_df)} ({len(train_df)/n:.1%}), "
          f"Val: {len(val_df)} ({len(val_df)/n:.1%}), "
          f"Test: {len(test_df)} ({len(test_df)/n:.1%})")
    
    return train_df, val_df, test_df

def create_time_segments(train_df: pd.DataFrame, segment_length: int = 80, standardize: bool = False) -> tuple:
    """
    Splits each stock's time series in the training set into disjoint, time-ordered segments
    of fixed length.
    
    :param train_df: DataFrame with dates as index and stock tickers as columns
    :param segment_length: Length of each segment (default: 80 trading days)
    :param standardize: If True, each segment is normalized by subtracting the mean and 
                        dividing by the standard deviation (default: False)
    :return: Tuple of (segments_df, metadata_df) where segments_df has standardized timestep 
             indices (0 to segment_length-1) containing all segments
    """
    segments_list = []
    segment_ids = []
    tickers = []
    
    for ticker in train_df.columns:
        ticker_data = train_df[ticker].dropna()
        
        if len(ticker_data) < segment_length:
            print(f"Skipping {ticker}: insufficient data (only {len(ticker_data)} points)")
            continue
            
        num_segments = len(ticker_data) // segment_length
        
        if num_segments == 0:
            continue
            
        # create segments - vectorized approach for better performance
        segments = np.array_split(ticker_data.values[:num_segments*segment_length], num_segments)
        
        for i, segment in enumerate(segments):
            if len(segment) == segment_length:
                if standardize:
                    segment_mean = np.mean(segment)
                    segment_std = np.std(segment)
                    if segment_std > 0:
                        segment = (segment - segment_mean) / segment_std
                    else:
                        segment = segment - segment_mean
                
                segments_list.append(segment)
                segment_ids.append(f"{ticker}_segment_{i+1}")
                tickers.append(ticker)
    
    segments_df = pd.DataFrame(segments_list, index=segment_ids).T
    segments_df.index = range(segment_length)
    segments_df.columns.name = 'segment_id'
    
    metadata = pd.DataFrame({'ticker': tickers}, index=segment_ids)
    
    print(f"Created {segments_df.shape[1]} segments from {len(set(tickers))}/{len(train_df.columns)} tickers")
    print(f"Each segment has {segment_length} timesteps")
    if standardize:
        print("Segments were standardized (zero mean, unit variance)")
    
    return segments_df, metadata

def save_segments_to_parquet(segments_df: pd.DataFrame, metadata: pd.DataFrame, 
                         segments_filename: str, metadata_filename: str) -> None:
    """
    Saves the segments DataFrame and metadata to Parquet files.
    
    :param segments_df: DataFrame with segments data
    :param metadata: DataFrame with segment metadata
    :param segments_filename: Filename for segments
    :param metadata_filename: Filename for metadata
    """    
    segments_df.to_parquet(segments_filename)
    metadata.to_parquet(metadata_filename)
    print(f"Segments saved to {segments_filename}")
    print(f"Metadata saved to {metadata_filename}")

def save_data_to_parquet(data: pd.DataFrame, filename: str) -> None:
    """
    Saves the data to a Parquet file.
    
    :param data: DataFrame to save
    :param filename: Name of the Parquet file
    """
    data.to_parquet(filename)
    print(f"Data saved to {filename}")

def load_data_from_parquet(filename: str) -> pd.DataFrame:
    """
    Loads historical stock data from a Parquet file.
    
    :param filename: Path to the Parquet file
    :return: DataFrame with dates as index and tickers as columns
    """
    try:
        data = pd.read_parquet(filename)
        print(f"Successfully loaded data from {filename}")
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        return data
    except Exception as e:
        raise ValueError(f"Failed to load data from {filename}: {e}")


if __name__ == "__main__":
    main_dir = 'data/data_storage/sp500_parquets'
    input_filename = os.path.join(main_dir, 'sp500_historical_data.parquet')
    train_filename = os.path.join(main_dir, 'sp500_train.parquet')
    val_filename = os.path.join(main_dir, 'sp500_val.parquet')
    test_filename = os.path.join(main_dir, 'sp500_test.parquet')
    
    # option to download new data or use existing, set to True to use existing data
    use_existing_data = True 
    
    if use_existing_data:
        data = load_data_from_parquet(input_filename)
        if data is None:
            raise ValueError("Failed to load existing data. Please check the file path.")
    else:
        start_date = '2000-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        data_type = 'Close'  
        
        print("Getting S&P 500 tickers...")
        tickers = get_sp500_tickers()
        print(f"Found {len(tickers)} tickers")
    
        data = download_sp500_data(tickers, start_date, end_date, data_type)
        save_data_to_parquet(data, input_filename)
    
    train_df, val_df, test_df = split_time_series_data(data)
    
    save_data_to_parquet(train_df, train_filename)
    save_data_to_parquet(val_df, val_filename)
    save_data_to_parquet(test_df, test_filename)
    
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

    segments_filename = os.path.join(main_dir, 'sp500_segments.parquet')
    metadata_filename = os.path.join(main_dir, 'sp500_segments_metadata.parquet')
    
    print("\nCreating time segments from training data...")
    segments_df, metadata = create_time_segments(train_df, segment_length=80, standardize=True)
    
    save_segments_to_parquet(segments_df, metadata, segments_filename, metadata_filename)