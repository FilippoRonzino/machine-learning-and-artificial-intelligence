import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_actual_vs_predicted(y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             percentage_predicted: float) -> None:
    """
    Plots the actual vs predicted values with a fill between the two lines.
    :param y_true: Actual values (1D array).
    :param y_pred: Predicted values (1D array).
    :param percentage_predicted: Percentage of predicted values over the total number of y_pred
    :return: None
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

def plot_model_predictions(model, dataloader, n_plots=5)-> None:
    """
    Plots actual vs predicted values from the model on `n_plots` random samples from the dataloader.
    It only plots the last `actual_prediction_len` steps of the output sequence.
    :param model: The trained model.
    :param dataloader: DataLoader containing the data.
    :param n_plots: Number of plots to generate.
    """
    model.eval()
    count = 0
    
    for batch in dataloader:
        x_batch, y_batch = batch  

        for i in range(x_batch.size(0)):
            if count >= n_plots:
                return

            x_sample = x_batch[i]                        
            y_sample = y_batch[i]                        
            x_input = x_sample.unsqueeze(0)             

            with torch.no_grad():
                y_pred_full = model(x_input)             

            y_pred_future = y_pred_full[0, -model.actual_prediction_len:, :]
            y_true_future = y_sample[-model.actual_prediction_len:, :]

            x_part = x_sample.flatten().cpu().numpy()
            y_true_future_flat = y_true_future.flatten().cpu().numpy()
            y_pred_future_flat = y_pred_future.flatten().cpu().numpy()
            
            y_true_combined = np.concatenate((x_part, y_true_future_flat))
            y_pred_combined = np.concatenate((x_part, y_pred_future_flat))

            plot_actual_vs_predicted(
                y_true_combined, 
                y_pred_combined, 
                percentage_predicted=0.25
            )

            count += 1