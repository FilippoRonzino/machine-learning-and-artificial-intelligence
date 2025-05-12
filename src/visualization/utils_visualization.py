import glob
import os
import random
import re
from enum import Enum
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset

from models.cnn_numerical import CNNTimeSeriesPredictor
from models.cnn_visual import CNN_Autoencoder, ImageTimeSeriesDatasetSingleFolder
from models.utils_models import extract_loss_from_event, load_and_prepare_data


class DatasetType(Enum):
    ECG = "ecg"
    OU = "ou"
    SP500 = "sp500"
    HARMONIC = "harmonic"

class ModelType(Enum):
    CNN_NUMERICAL = "cnn_numerical"
    CNN_VISUAL = "cnn_visual"


def find_latest_checkpoint(checkpoint_folder: str) -> str:
    """
    Find the checkpoint file with the highest epoch number in the folder.
    
    :param checkpoint_folder: Folder containing the checkpoint files
    :return: Name of the latest checkpoint file
    """
    checkpoint_path = os.path.join("src/models", checkpoint_folder)
    checkpoint_files = glob.glob(os.path.join(checkpoint_path, "*.ckpt"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
    
    epoch_pattern = re.compile(r'epoch=epoch=(\d+)')
    
    def get_epoch(filename):
        match = epoch_pattern.search(filename)
        if match:
            return int(match.group(1))
        return -1
    
    latest_checkpoint = max(checkpoint_files, key=get_epoch)
    print(f"Latest checkpoint found: {latest_checkpoint}")
    return os.path.basename(latest_checkpoint)

def find_event_file(model_type: Union[ModelType, str], dataset_type: Union[DatasetType, str]) -> str:
    """
    Find the most recent event file for the given model and dataset type.
    
    :param model_type: ModelType enum or string representing model type
    :param dataset_type: DatasetType enum or string representing dataset type
    :return: Path to the most recent event file
    """
    model_name = model_type.value if hasattr(model_type, 'value') else model_type
    dataset_name = dataset_type.value if hasattr(dataset_type, 'value') else dataset_type
    
    log_path = f"src/models/lightning_logs/{model_name}_{dataset_name}"
    
    # Find the highest version number
    versions = glob.glob(os.path.join(log_path, "version_*"))
    if not versions:
        raise FileNotFoundError(f"No version folders found in {log_path}")
    
    latest_version = max(versions, key=lambda x: int(x.split('_')[-1]))
    
    event_files = glob.glob(os.path.join(latest_version, "events.out.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {latest_version}")
    
    most_recent = sorted(event_files)[-1]
    print(f"Most recent event file found: {most_recent}")
    return most_recent


def plot_actual_vs_predicted(y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             percentage_predicted: float) -> None:
    """
    Plots the actual vs predicted values with a fill between the two lines.
    :param y_true: Actual values (1D array).
    :param y_pred: Predicted values (1D array).
    :param percentage_predicted: Percentage of predicted values over the total number of y_pred
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Both y_true and y_pred must be one-dimensional.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of elements.")
    
    if not (0 <= percentage_predicted <= 1):
        raise ValueError("percentage_predicted must be between 0 and 100.")
    
    num_points = y_true.shape[0]
    num_predicted_points = int(percentage_predicted  * num_points)
    
    predicted_mask = np.zeros(num_points, dtype=bool)
    if num_predicted_points > 0:
        predicted_mask[-num_predicted_points:] = True
    
    x_vals = np.arange(num_points)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(x_vals, y_true, label="Actual", color="green", linestyle="-")
    
    plt.plot(x_vals, np.where(predicted_mask, y_pred, np.nan), 
             label="Predicted", color="red", linestyle=":")
    
    plt.fill_between(x_vals, y_true, np.where(predicted_mask, y_pred, np.nan),
                     color="gray", alpha=0.5)
    
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Actual vs Predicted Values")
    plt.legend()
    plt.grid(True)
    
    plt.show()

def plot_predictions_numerical_model(model: CNNTimeSeriesPredictor, 
                                     dataset: torch.utils.data.TensorDataset, 
                                     n_plots: int = 5, all_predictions: bool = False) -> None:
    """
    Plots actual vs predicted values from the model on `n_plots` random samples from the dataset.
    It only plots the last `actual_prediction_len` steps of the output sequence.
    
    :param model: The trained model (CNNTimeSeriesPredictor).
    :param dataset: Dataset containing the data (TensorDataset).
    :param n_plots: Number of plots to generate.
    """
    model.eval()
    indices = random.sample(range(len(dataset)), n_plots)

    for idx in indices:
        x_sample, y_sample = dataset[idx]
        x_input = x_sample.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            y_pred_full = model(x_input.to(model.device))  # Move input to device, output is [1, T, D]

        
        y_pred_future = y_pred_full[0, -model.actual_prediction_len:, :].cpu()
        
        y_true_future = y_sample[-model.actual_prediction_len:, :].cpu()

        x_part = x_sample.flatten().cpu().numpy()
        y_true_future_flat = y_true_future.flatten().numpy()
        y_pred_future_flat = y_pred_future.flatten().numpy()

        y_true_combined = np.concatenate((x_part, y_true_future_flat))
        y_pred_combined = np.concatenate((x_part, y_pred_future_flat))
        if all_predictions:
            y_pred_future = y_pred_full.cpu().flatten().numpy()
            y_pred_future = np.concatenate((np.full(20, np.nan), y_pred_future))
            plot_actual_vs_predicted(
                y_true_combined,
                y_pred_future,
                percentage_predicted=1
            )
        else:
            plot_actual_vs_predicted(
                y_true_combined,
                y_pred_combined,
                percentage_predicted=0.25)

def plot_predictions_visual_model(model: CNN_Autoencoder, 
                                  dataset: ImageTimeSeriesDatasetSingleFolder, 
                                  n_images: int = 5) -> None:
    """
    Plot model predictions (visual models) against ground truth.

    :param model: Trained model (CNN_Autoencoder).
    :param dataset: Dataset to visualize predictions from (ImageTimeSeriesDatasetSingleFolder).
    :param n_images: Number of images to visualize.
    """
    model.eval()
    indices = random.sample(range(len(dataset)), n_images)
    fig, axes = plt.subplots(n_images, 3, figsize=(12, 3 * n_images))

    # For a single plot, convert to 2D array with shape (1, 3)
    if n_images == 1:
        print("Only one image to plot, converting axes to 2D array.")
        axes = np.array([axes])

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

def plot_predictions(model: CNNTimeSeriesPredictor | CNN_Autoencoder, 
                     dataset: torch.utils.data.TensorDataset | ImageTimeSeriesDatasetSingleFolder, 
                     n_plots: int = 5, all_predictions: bool = False) -> None:
    """
    Plot predictions from the model on the dataset.
    
    :param model: The trained model (CNNTimeSeriesPredictor or CNN_Autoencoder).
    :param dataset: Dataset containing the data.
    :param n_plots: Number of plots to generate.
    """
    if isinstance(model, CNNTimeSeriesPredictor):
        plot_predictions_numerical_model(model, dataset, n_plots, all_predictions)
    elif isinstance(model, CNN_Autoencoder):
        plot_predictions_visual_model(model, dataset, n_plots)
    else:
        raise ValueError("Unsupported model type. Please provide a CNNTimeSeriesPredictor or CNN_Autoencoder.")

def plot_train_val_loss(train_data: tuple[list[int], list[float]], 
                       val_data: tuple[list[int], list[float]], 
                       title: str = "Training and Validation Loss") -> None:
    """
    Plots training and validation loss curves.
    
    :param train_data: Tuple containing (step_indices, training_loss_values)
    :param val_data: Tuple containing (step_indices, validation_loss_values)
    :param title: Title for the plot
    """
    train_steps, train_values = train_data
    val_steps, val_values = val_data
    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_values, label='Training Loss')
    plt.plot(val_steps, val_values, label='Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()

def run_numerical_evaluation(checkpoint_folder: str, 
                            ckpt_file: str, 
                            parquet_path: str, 
                            event_file: str, 
                            base_dir: str = "src/models", 
                            prediction_percentage: float = 0.25, 
                            n_plots: int = 1, 
                            all_predictions: bool = True) -> None:
    """
    Run evaluation for numerical models.
    
    :param checkpoint_folder: Path to folder containing model checkpoints
    :param ckpt_file: Checkpoint filename
    :param parquet_path: Path to test data in parquet format
    :param event_file: Path to TensorBoard event file with training history
    :param base_dir: Base directory for models
    :param prediction_percentage: Percentage of sequence to predict
    :param n_plots: Number of plots to generate
    :param all_predictions: Whether to show all predictions or only future ones
    """
    model = CNNTimeSeriesPredictor.load_from_checkpoint(
        os.path.join(base_dir, checkpoint_folder, ckpt_file),
        map_location=torch.device("cpu")
    )
    model.eval()

    X_test, y_test = load_and_prepare_data(parquet_path, prediction_percentage=prediction_percentage)
    val_dataset = TensorDataset(torch.stack(X_test), torch.stack(y_test))
    plot_predictions(model, val_dataset, n_plots=n_plots, all_predictions=all_predictions)

    train_data, val_data = extract_loss_from_event(event_file, train_tag='train_loss', val_tag='val_loss')
    plot_train_val_loss(train_data, val_data, title="Training and Validation Loss")

def run_visual_evaluation(base_dir: str, 
                         checkpoint_folder: str, 
                         ckpt_file: str, 
                         data_dir: str, 
                         event_file: str, 
                         prediction_percentage: float = 0.25, 
                         n_plots: int = 5) -> None:
    """
    Run evaluation for visual models.
    
    :param base_dir: Base directory for models
    :param checkpoint_folder: Path to folder containing model checkpoints
    :param ckpt_file: Checkpoint filename
    :param data_dir: Directory containing test images
    :param event_file: Path to TensorBoard event file with training history
    :param prediction_percentage: Percentage of sequence to predict
    :param n_plots: Number of plots to generate
    """
    model = CNN_Autoencoder.load_from_checkpoint(
        os.path.join(base_dir, checkpoint_folder, ckpt_file),
        map_location=torch.device("cpu")
    )
    model.eval()

    dataset = ImageTimeSeriesDatasetSingleFolder(data_dir, prediction_percentage=prediction_percentage)
    plot_predictions(model, dataset, n_plots=n_plots)

    train_data, val_data = extract_loss_from_event(event_file, train_tag='train_loss', val_tag='val_loss')
    plot_train_val_loss(train_data, val_data, title="Training and Validation Loss")

def automated_evaluation(model_type: ModelType, 
                        dataset_type: DatasetType, 
                        base_dir: str = "src/models", 
                        prediction_percentage: float = 0.25, 
                        n_plots: int = 3, 
                        all_predictions: bool = True) -> None:
    """
    Automated evaluation that finds the latest checkpoint and runs evaluation.
    
    :param model_type: ModelType enum (CNN_NUMERICAL or CNN_VISUAL)
    :param dataset_type: DatasetType enum (ECG, OU, SP500, HARMONIC)
    :param base_dir: Base directory for models
    :param prediction_percentage: Percentage of the sequence to predict
    :param n_plots: Number of plots to generate
    :param all_predictions: Whether to show all predictions or only future ones
    """
    dataset_name = dataset_type.value
    model_name = model_type.value
    
    checkpoint_folder = f"checkpoints/{model_name}_{dataset_name}"
    
    ckpt_file = find_latest_checkpoint(checkpoint_folder)
    event_file = find_event_file(model_type, dataset_type)
    
    if model_type == ModelType.CNN_NUMERICAL:
        # Special case for OU and HARMONIC numerical datasets
        if dataset_type in [DatasetType.HARMONIC, DatasetType.OU]:
            parquet_path = f"data/data_storage/harmonic_ou_parquets/test_{dataset_name}.parquet"
        else:
            parquet_path = f"data/data_storage/{dataset_name}_parquets/test_{dataset_name}.parquet"
        
        run_numerical_evaluation(
            checkpoint_folder=checkpoint_folder,
            ckpt_file=ckpt_file,
            parquet_path=parquet_path,
            event_file=event_file,
            base_dir=base_dir,
            prediction_percentage=prediction_percentage,
            n_plots=n_plots,
            all_predictions=all_predictions
        )
    
    elif model_type == ModelType.CNN_VISUAL:
        # Path to test data for visual models
        data_dir = f"data/images/{dataset_name}/test"
        
        run_visual_evaluation(
            base_dir=base_dir,
            checkpoint_folder=checkpoint_folder,
            ckpt_file=ckpt_file,
            data_dir=data_dir,
            event_file=event_file,
            prediction_percentage=prediction_percentage,
            n_plots=n_plots
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
if __name__ == "__main__":
    automated_evaluation(
        model_type=ModelType.CNN_NUMERICAL,
        dataset_type=DatasetType.ECG,
        n_plots=1,
        all_predictions=True
    )
    
    automated_evaluation(
        model_type=ModelType.CNN_VISUAL,
        dataset_type=DatasetType.SP500,
        n_plots=1,
        all_predictions=True
    )