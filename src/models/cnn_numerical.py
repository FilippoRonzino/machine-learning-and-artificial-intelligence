import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from models.utils_models import get_device, load_and_prepare_data
from visualization.utils_visualization import plot_predictions


class CNNTimeSeriesPredictor(pl.LightningModule):
    def __init__(self, input_features, input_seq_len, output_features,
                 model_output_seq_len, actual_prediction_len,
                 cnn_layers = 3, kernel_size = 3, 
                 base_filters = 32, fc_size = 128, lr = 0.001):
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
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
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
        plt.ylabel("Loss (MSE on last {} steps)".format(self.actual_prediction_len))
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    

# if __name__ == "__main__":
#     device = get_device()

#     model = CNNTimeSeriesPredictor(
#         input_features = 1,
#         input_seq_len = 60,
#         output_features = 1,
#         model_output_seq_len = 60,  
#         actual_prediction_len = 20,     
#         cnn_layers = 3,
#         kernel_size = 3,
#         base_filters = 32,
#         fc_size = 128,
#         lr = 0.001
#     )

#     model = model.to(device)

#     file_path_train = "data/data_storage/harmonic_ou_parquets/train_harmonic.parquet"
#     file_path_test = "data/data_storage/harmonic_ou_parquets/test_harmonic.parquet"
    
#     X_train, y_train = load_and_prepare_data(file_path_train, prediction_percentage=0.25)
#     X_test, y_test = load_and_prepare_data(file_path_test, prediction_percentage=0.25)

#     print("X_train sample shape:", X_train[0].shape)
#     print("y_train sample shape:", y_train[0].shape)
#     print(len(X_train), "samples loaded for training")
#     print(len(X_test), "samples loaded for testing")

#     train_dataset = TensorDataset(torch.stack(X_train), torch.stack(y_train))
#     val_dataset = TensorDataset(torch.stack(X_test), torch.stack(y_test))

#     train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True, num_workers=7)
#     val_loader = DataLoader(val_dataset, batch_size = 32, num_workers=7)

#     print("Setting up trainer...")
#     loss_tracker = LossTrackerCallback()
#     trainer = pl.Trainer(
#         max_epochs = 1,
#         enable_checkpointing = False,
#         logger = False,
#         accelerator = "auto",
#         callbacks = [loss_tracker],
#         # num_sanity_val_steps = 0
#     )

#     print("Starting training...")
#     trainer.fit(model, train_loader, val_loader)
#     print("Training finished.")

#     print("Plotting losses...")
#     loss_tracker.plot_losses()

#     plot_predictions(model, val_dataset, n_plots=5)
