import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time
import os
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

def save_data_to_csv(data: pd.DataFrame, filename: str) -> None:
    """
    Saves the data to a CSV file.
    
    :param data: DataFrame to save
    :param filename: Name of the CSV file
    """
    data.to_csv(filename)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    # Parameters
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    data_type = 'Close'  # Options: Open, High, Low, Close, Adj Close, Volume
    output_filename = 'data/sp500_csvs/sp500_historical_data.csv'
    
    # Get S&P 500 tickers
    print("Getting S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} tickers")
    
    # Download data
    data = download_sp500_data(tickers, start_date, end_date, data_type)
    
    # Save to CSV
    save_data_to_csv(data, output_filename)
    
    # Print summary statistics
    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Number of tickers with data: {len(data.columns)}")
    print(f"Percentage of missing values: {data.isna().mean().mean() * 100:.2f}%")