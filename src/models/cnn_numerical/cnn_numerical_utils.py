import numpy as np
from data.data_loader import  load_time_series_parquet

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
    file_path = "/Users/giuseppeiannone/machine-learning-and-artificial-intelligence/data/data_storage/ecg_parquets/test_ecg.parquet"
    prediction_percentage = 25

    X, y = load_and_prepare_data(file_path, prediction_percentage)
    print("X shape:", X[0].shape)
    print("y shape:", y[0].shape)
    print(len(X), "samples loaded")
