import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
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
    X_all = [torch.tensor(X).float() for X in X_all]
    y_all = [torch.tensor(y).float().unsqueeze(-1) for y in y_all]
    return X_all, y_all

def plot_predictions_visual_model(model, dataset, n_images=5):
    """
    Plot model predictions (visual models) against ground truth.

    :param model: Trained model (CNN_Autoencoder).
    :param dataset: Dataset to visualize predictions from.
    :param n_images: Number of images to visualize.
    """
    model.eval()
    indices = random.sample(range(len(dataset)), n_images)
    fig, axes = plt.subplots(n_images, 3, figsize=(12, 3 * n_images))

    for i, idx in enumerate(indices):
        input_tensor, target_tensor = dataset[idx]
        input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            reconstructed = model(input_tensor.to(model.device)).cpu().squeeze()

        input_image = input_tensor.squeeze().numpy()
        target_image = target_tensor.squeeze().numpy()
        recon_image = reconstructed.numpy()

        axes[i, 0].imshow(input_image, cmap="gray")
        axes[i, 0].set_title("Input")
        axes[i, 1].imshow(target_image, cmap="gray")
        axes[i, 1].set_title("Expected Output")
        axes[i, 2].imshow(recon_image, cmap="gray")
        axes[i, 2].set_title("Reconstructed Output")

        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()

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
