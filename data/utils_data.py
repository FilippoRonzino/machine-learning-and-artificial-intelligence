import yfinance as yf
import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time

def get_sp500_tickers():
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

def download_data(ticker: str, start_date: datetime, end_date: datetime):
    """
    Downloads historical data for a single ticker.

    :param ticker: Ticker symbol
    :param start_date: Start date for historical data
    :param end_date: End date for historical data
    :return: DataFrame with historical data
    """
    try:
        df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        return df
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return pd.DataFrame()

def extract_close_prices(df: pd.DataFrame, ticker: str):
    """
    Extracts the 'Close' column from a DataFrame, renames it to the ticker, 
    and sets the date as the index.

    :param df: DataFrame with historical data
    :param ticker: Ticker symbol
    :return: DataFrame with 'Close' prices and date as index
    """
    if 'Close' in df.columns:
        close_df = df[['Close']].copy() 
        close_df.rename(columns={'Close': ticker}, inplace=True)  
        close_df = close_df.set_index(df.index)  
        return close_df
    else:
        return pd.DataFrame()

def build_close_price_df(tickers: list, start_date: datetime, end_date: datetime):
    """
    Builds a DataFrame with 'Close' prices for all tickers, with dates as index and tickers as columns.

    :param tickers: List of ticker symbols
    :param start_date: Start date for historical data
    :param end_date: End date for historical data
    :return: DataFrame with 'Close' prices
    """
    all_closes = pd.DataFrame()

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = download_data(ticker, start_date, end_date)
        if not df.empty:
            close_df = extract_close_prices(df, ticker)
            all_closes = pd.concat([all_closes, close_df], axis=1)
        else:
            print(f"Warning: No data for {ticker}")
        time.sleep(0.5)  # avoid rate limiting
    
    return all_closes

if __name__ == "__main__":
    tickers = get_sp500_tickers()
    start_date = datetime(2000, 1, 1)
    end_date = datetime.now()

    close_prices_df = build_close_price_df(tickers, start_date, end_date)

    print("\nFinal merged Close price DataFrame:")
    print(close_prices_df.head())

    # Save to disk
    close_prices_df.to_csv("sp500_close_prices.csv")
