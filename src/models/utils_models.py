import os
import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing import event_accumulator
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

def prepare_data(series, prediction_percentage=0.25):
    total_steps = len(series)
    input_steps = int(total_steps * (1 - prediction_percentage))

    X = series[:input_steps].reshape((1, input_steps, 1))
    y = series[-input_steps:].reshape((1, input_steps))  

    return X, y

def load_and_prepare_data(file_path, prediction_percentage=0.25):
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
    X_all = [torch.tensor(X).float() for X in X_all]
    y_all = [torch.tensor(y).float().unsqueeze(-1) for y in y_all]
    return X_all, y_all

def get_device():
    """
    Get the appropriate device for PyTorch operations.
    
    :return: A torch.device object representing the device to use.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device

def get_val_loss(filename):
    try:
        # Extract the value after "val_loss="
        val_loss_str = filename.split('val_loss=')[-1].split('-')[0].split('.ckpt')[0]
        return float(val_loss_str)
    except (ValueError, IndexError):
        return float('inf')  
    
def extract_loss_from_event(event_file, train_tag='train_loss', val_tag='val_loss'):
    """
    Extracts training and validation loss from a TensorBoard event file.
    
    :param event_file: Path to the TensorBoard event file (e.g., "events.out.tfevents...").
    :param train_tag: Tag for training loss.
    :param val_tag: Tag for validation loss.
    :return: Tuple of (train_steps, train_values), (val_steps, val_values)
    """
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    print("Available tags:", ea.Tags())
    
    train_loss = ea.Scalars(train_tag)
    val_loss = ea.Scalars(val_tag)
    
    train_steps = [x.step for x in train_loss]
    train_values = [x.value for x in train_loss]
    val_steps = [x.step for x in val_loss]
    val_values = [x.value for x in val_loss]
    
    return (train_steps, train_values), (val_steps, val_values)

def smooth_data(values, factor):
    """
    Apply exponential moving average smoothing to values.

    :param values: List of values to smooth.
    :param factor: Smoothing factor (0 < factor < 1). A higher factor means less smoothing.
    :return: List of smoothed values.
    """
    if factor <= 0:
        return values
    
    smoothed = []
    last = values[0]
    for value in values:
        smoothed_val = last * factor + (1 - factor) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


if __name__ == "__main__":
    file_path = "data/data_storage/ecg_parquets/test_ecg.parquet" 
    row_index = 21 
    
    series = extract_series_from_parquet(file_path, row_index)
    is_stationary = check_stationarity(series)

    print(f"Is the time series stationary? {'Yes' if is_stationary else 'No'}")

    prediction_percentage = 0.25

    X, y = load_and_prepare_data(file_path, prediction_percentage)
    print("X shape:", X[0].shape)
    print("y shape:", y[0].shape)
    print(f"{len(X)} samples loaded for training")
