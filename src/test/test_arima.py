import numpy as np
import pandas as pd

from data.data_loader import extract_series_from_parquet
from models.arima import AutoARIMA
from visualization.utils_visualization import plot_actual_vs_predicted


def test_arima_on_dataset(data_path, dataset_name, column_index=0, seasonal=False):
    """
    Test AutoARIMA on the given dataset and plot the results
    
    :param data_path: Path to the dataset
    :param dataset_name: Name of the dataset for logging
    :column_index: Index of the column to use for time series
    :param seasonal: Whether to use seasonal ARIMA
    :return: A dictionary containing model, metrics, predictions, train_data, and test_data
    """
    data = extract_series_from_parquet(data_path, column_index)
    print(f"Using {dataset_name} data of length {len(data)}")
    
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    print(f"Training ARIMA on {len(train_data)} samples, testing on {len(test_data)} samples")
    
    auto_model = AutoARIMA(seasonal=seasonal)
    auto_model.fit(train_data, check_stationarity_first=True)
    auto_model.summary()
    
    predictions = auto_model.predict(n_periods=len(test_data))
    
    metrics = auto_model.evaluate(test_data)
    print(f"{dataset_name} AutoARIMA - MASE: {metrics['MASE']:.4f}, SMAPE: {metrics['SMAPE']:.4f}")
    
    combined_series = pd.Series(np.concatenate((train_data, predictions)), 
                               index=np.arange(len(train_data) + len(predictions)))
    
    plot_actual_vs_predicted(
        y_true=data,
        y_pred=combined_series,
        percentage_predicted=20,
    )
    
    return {
        "model": auto_model,
        "metrics": metrics,
        "predictions": predictions,
        "train_data": train_data,
        "test_data": test_data
    }

if __name__ == "__main__":
    # Test ECG data
    ecg_results = test_arima_on_dataset(
        data_path='data/data_storage/ecg_parquets/test_ecg.parquet',
        dataset_name='ECG'
    )

    # Test Harmonic data
    harmonic_results = test_arima_on_dataset(
        data_path='data/data_storage/harmonic_ou_parquets/test_harmonic.parquet',
        dataset_name='Harmonic'
    )

    # Test OU data
    ou_results = test_arima_on_dataset(
        data_path='data/data_storage/harmonic_ou_parquets/test_ou.parquet',
        dataset_name='OU'
    )

    # Test S&P 500 data
    sp500_results = test_arima_on_dataset(
        data_path='data/data_storage/sp500_parquets/test_sp500.parquet',
        dataset_name='S&P 500',
        column_index=35
    )