
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def load_time_series_parquet(file_path: str) -> pd.DataFrame:
    """
    Load a time series from a parquet file.

    :param file_path: Path to the parquet file.
    :return: A pandas DataFrame containing the time series data.
    """
    df = pd.read_parquet(file_path)
    return df

def extract_series_from_parquet(file_path: str, row_index: int) -> pd.Series:
    """
    Extract a time series from a row in a parquet file.
    
    :file_path: Path to the parquet file.
    :row_index: Index of the row to extract.
    :return: A pandas Series representing the time series.
    """
    df = pd.read_parquet(file_path)
    return df.iloc[row_index]

def check_stationarity(row: pd.Series) -> bool:
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test.

    :param row: A pandas Series representing the time series.
    :return: True if the time series is stationary, False otherwise, p-value <= 0.05.
    """

    result = adfuller(row)
    return result[1] <= 0.05 

if __name__ == "__main__":
    # Example usage
    file_path = "data/data_storage/ecg_parquets/test_ecg.parquet" 
    row_index = 21  # Change this to the desired row index

    series = extract_series_from_parquet(file_path, row_index)
    is_stationary = check_stationarity(series)

    print(f"Is the time series stationary? {'Yes' if is_stationary else 'No'}")