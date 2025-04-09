import matplotlib.pyplot as plt
import numpy as np

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
    
    if not (0 <= percentage_predicted <= 100):
        raise ValueError("percentage_predicted must be between 0 and 100.")
    
    num_points = y_true.shape[0]
    num_predicted_points = int((percentage_predicted / 100) * num_points)
    
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

