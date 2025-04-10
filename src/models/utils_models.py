import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

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

def check_stationarity(series):
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test.
    
    :param series: A pandas Series representing the time series.
    :return: Tuple of (is_stationary, p_value, suggested_differencing_order)
    """
    result = adfuller(series)
    p_value = result[1]
    
    is_stationary = p_value < 0.05
    
    # heuristic for suggesting differencing order
    suggested_d = 0
    if not is_stationary:
        # Try first difference
        if isinstance(series, pd.Series):
            first_diff = series.diff().dropna()
        else:
            first_diff = pd.Series(series).diff().dropna()
            
        first_diff_result = adfuller(first_diff)
        if first_diff_result[1] < 0.05:
            suggested_d = 1
        else:
            # Try second difference
            second_diff = first_diff.diff().dropna()
            second_diff_result = adfuller(second_diff)
            if second_diff_result[1] < 0.05:
                suggested_d = 2
            else:
                suggested_d = 1  # Default to 1 if we're not sure
    
    return is_stationary, p_value, suggested_d

def prepare_data(series, prediction_percentage=25):
    total_steps = len(series)
    input_steps = int(total_steps * ((100 - prediction_percentage) / 100))

    X = series[:input_steps].reshape((1, input_steps, 1))
    y = series[-input_steps:].reshape((1, input_steps))  

    return X, y

def load_and_prepare_data(file_path, prediction_percentage=25):
    df = load_time_series_parquet(file_path)
    if df is None:
        raise ValueError(f"Failed to load data from {file_path}")
    
    X_all = []
    y_all = []

    for _, row in df.iterrows():
        series = row.values  # Full time series (80 steps)
        X, y = prepare_data(series, prediction_percentage)
        X_all.append(X)
        y_all.append(y)
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    return X_all, y_all


if __name__ == "__main__":
    file_path = "data/data_storage/ecg_parquets/test_ecg.parquet" 
    row_index = 21 
    
    series = extract_series_from_parquet(file_path, row_index)
    is_stationary = check_stationarity(series)

    print(f"Is the time series stationary? {'Yes' if is_stationary else 'No'}")

    prediction_percentage = 25

    X, y = load_and_prepare_data(file_path, prediction_percentage)
    print("X shape:", X[0].shape)
    print("y shape:", y[0].shape)
    print(f"{len(X)} samples loaded for training")
