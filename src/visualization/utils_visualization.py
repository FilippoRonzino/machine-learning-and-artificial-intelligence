import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.cnn_numerical import CNNTimeSeriesPredictor
from models.cnn_visual import CNN_Autoencoder, ImageTimeSeriesDatasetSingleFolder


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