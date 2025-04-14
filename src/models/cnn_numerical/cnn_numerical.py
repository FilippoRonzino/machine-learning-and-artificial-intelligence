from models.utils_models import load_and_prepare_data
from visualization.utils_visualization import plot_actual_vs_predicted
import pytorch_lightning as pl 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
TOTAL_SEQUENCE_LENGTH = 80
INPUT_SEQUENCE_LENGTH = 60  
MODEL_OUTPUT_SEQUENCE_LENGTH = 60  
ACTUAL_PREDICTION_LENGTH = TOTAL_SEQUENCE_LENGTH - INPUT_SEQUENCE_LENGTH  

if INPUT_SEQUENCE_LENGTH + ACTUAL_PREDICTION_LENGTH != TOTAL_SEQUENCE_LENGTH:
    raise ValueError("Inconsistency: INPUT_SEQUENCE_LENGTH + ACTUAL_PREDICTION_LENGTH must equal TOTAL_SEQUENCE_LENGTH")
if MODEL_OUTPUT_SEQUENCE_LENGTH != TOTAL_SEQUENCE_LENGTH - (TOTAL_SEQUENCE_LENGTH - MODEL_OUTPUT_SEQUENCE_LENGTH):
    target_start_index = TOTAL_SEQUENCE_LENGTH - MODEL_OUTPUT_SEQUENCE_LENGTH
    if target_start_index < 0:
        raise ValueError("MODEL_OUTPUT_SEQUENCE_LENGTH cannot be greater than TOTAL_SEQUENCE_LENGTH")
    print(f"Model output corresponds to original sequence steps {target_start_index} to {TOTAL_SEQUENCE_LENGTH-1}")

INPUT_FEATURES = 1          
OUTPUT_FEATURES = 1        
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 2

# CNN parameters
CNN_LAYERS = 3              # Number of convolutional layers
KERNEL_SIZE = 3             # Size of convolutional kernels
BASE_FILTERS = 32           # Number of filters in first conv layer
FC_SIZE = 128               # Size of fully connected layer

# --- CNN Model ---
class CNNTimeSeriesPredictor(pl.LightningModule):
    def __init__(self, input_features, input_seq_len, output_features,
                 model_output_seq_len, actual_prediction_len,
                 cnn_layers=CNN_LAYERS, kernel_size=KERNEL_SIZE, 
                 base_filters=BASE_FILTERS, fc_size=FC_SIZE, lr=LEARNING_RATE):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_features = input_features
        self.input_seq_len = input_seq_len
        self.output_features = output_features
        self.model_output_seq_len = model_output_seq_len
        self.actual_prediction_len = actual_prediction_len
        self.cnn_layers = cnn_layers
        self.kernel_size = kernel_size
        self.base_filters = base_filters
        self.fc_size = fc_size
        self.lr = lr
        
        self.conv_layers = nn.ModuleList()
        
        self.conv_layers.append(
            nn.Conv1d(
                in_channels=input_features,
                out_channels=base_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2  # Same padding
            )
        )
        
        for i in range(1, cnn_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=base_filters * (2**(i-1)),
                    out_channels=base_filters * (2**i),
                    kernel_size=kernel_size,
                    padding=kernel_size // 2  # Same padding
                )
            )
        
        final_conv_channels = base_filters * (2**(cnn_layers-1))
        self.flatten_size = final_conv_channels * input_seq_len
        
        self.fc1 = nn.Linear(self.flatten_size, fc_size)
        self.fc2 = nn.Linear(fc_size, model_output_seq_len * output_features)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = x.permute(0, 2, 1)
        
        for conv in self.conv_layers:
            x = self.relu(conv(x))
        
        x = x.view(batch_size, -1)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        output = x.view(batch_size, self.model_output_seq_len, self.output_features)
        
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch  
        y_hat = self(x) 
        
        if y_hat.shape != y.shape:
            raise RuntimeError(f"Shape mismatch for loss: y_hat_relevant={y_hat.shape}, y_relevant={y.shape}")
        
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if y_hat.shape != y.shape:
            raise RuntimeError(f"Shape mismatch for loss: y_hat_relevant={y_hat.shape}, y_relevant={y.shape}")
        
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

class LossTrackerCallback(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.logged_metrics.get("val_loss")
        train_loss = trainer.logged_metrics.get("train_loss")
        if train_loss is not None:
            self.train_losses.append(train_loss.cpu().item())
        if val_loss is not None:
            self.val_losses.append(val_loss.cpu().item())

    def plot_losses(self):
        if not self.train_losses and not self.val_losses:
            print("No losses recorded to plot.")
            return
        plt.figure(figsize=(8, 5))
        if self.train_losses:
            plt.plot(self.train_losses, label="Training Loss")
        if self.val_losses:
            plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE on last {} steps)".format(ACTUAL_PREDICTION_LENGTH))
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    model = CNNTimeSeriesPredictor(
        input_features=INPUT_FEATURES,
        input_seq_len=INPUT_SEQUENCE_LENGTH,
        output_features=OUTPUT_FEATURES,
        model_output_seq_len=MODEL_OUTPUT_SEQUENCE_LENGTH,  
        actual_prediction_len=ACTUAL_PREDICTION_LENGTH,     
        cnn_layers=CNN_LAYERS,
        kernel_size=KERNEL_SIZE,
        base_filters=BASE_FILTERS,
        fc_size=FC_SIZE,
        lr=LEARNING_RATE
    )

    base_dir = "/Users/giuseppeiannone/machine-learning-and-artificial-intelligence"
    file_path_train = os.path.join(base_dir, "data", "data_storage", "harmonic_ou_parquets", "train_harmonic.parquet")
    file_path_test = os.path.join(base_dir, "data", "data_storage", "harmonic_ou_parquets", "test_harmonic.parquet")

    X_train, y_train = load_and_prepare_data(file_path_train, prediction_percentage=25)
    X_test, y_test = load_and_prepare_data(file_path_test, prediction_percentage=25)

    print("X_train sample shape:", X_train[0].shape)
    print("y_train sample shape:", y_train[0].shape)
    print(len(X_train), "samples loaded for training")
    print(len(X_test), "samples loaded for testing")

    train_dataset = TensorDataset(torch.stack(X_train), torch.stack(y_train))
    val_dataset = TensorDataset(torch.stack(X_test), torch.stack(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=7)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=7)

    print("Setting up trainer...")
    loss_tracker = LossTrackerCallback()
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        enable_checkpointing=False,
        logger=False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[loss_tracker],
        #num_sanity_val_steps=0
    )

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    print("Training finished.")

    print("Plotting losses...")
    loss_tracker.plot_losses()

    # --- Example Prediction (Adjusted Plotting) TO  CHANGE ---
    print("\n--- Example Prediction ---")
    model.eval().to('cpu')  # Set model to evaluation mode on CPU

    sample_idx = 0
    x_sample, y_sample = val_dataset[sample_idx]  # x_sample: [60, F], y_sample: [60, F]
    x_sample_batch = x_sample.unsqueeze(0)         # Add batch dimension: [1, 60, F]

    # Run inference in no_grad mode
    with torch.no_grad():
        y_pred_full = model(x_sample_batch)  # y_pred_full: [1, 60, F]

    # Extract the future predictions and true future values for the last ACTUAL_PREDICTION_LENGTH steps
    y_pred_future = y_pred_full[0, -ACTUAL_PREDICTION_LENGTH:, :]  # shape: [20, F]
    y_true_future = y_sample[-ACTUAL_PREDICTION_LENGTH:, :]         # shape: [20, F]

    # Flatten the historical input data to a 1D array
    x_part = x_sample.flatten().cpu().numpy()

    # Flatten the future true and predicted segments to 1D arrays
    y_true_future_flat = y_true_future.flatten().cpu().numpy()
    y_pred_future_flat = y_pred_future.flatten().cpu().numpy()

    # Concatenate the historical input with the future parts to create complete timelines
    y_true_combined = np.concatenate((x_part, y_true_future_flat))
    y_pred_combined = np.concatenate((x_part, y_pred_future_flat))


    plot_actual_vs_predicted(y_true_combined, y_pred_combined, 25)
