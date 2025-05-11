import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from models.utils_models import get_device
from visualization.utils_visualization import plot_predictions


class ImageTimeSeriesDatasetSingleFolder(Dataset):
    """
    Dataset for loading time series image data from a single folder.
    
    Loads images and splits them into input and target tensors where the target
    represents the future portion of the time series.
    """
    def __init__(self, source_dir, transform=None, prediction_percentage = 0.25):
        """
        Initialize the dataset.

        :param source_dir: Directory containing the image files.
        :param transform: Optional transform to be applied to the images.
        :param prediction_percentage: Percentage of the image width to be used as the target.
        """
        self.source_dir = source_dir
        self.transform = transform
        
        self.filenames = sorted([
            f for f in os.listdir(source_dir) 
            if os.path.isfile(os.path.join(source_dir, f)) 
               and f.lower().endswith('.png')
        ])
        if not self.filenames:
            raise FileNotFoundError(f"No PNG image files found in {source_dir}.")
        
        self.prediction_percentage = prediction_percentage

    def __len__(self):
        """
        :return: Number of images in the dataset.
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Load an image and split it into input and target tensors.

        :param idx: Index of the image to be loaded.
        :return: Tuple of input tensor and target tensor.
        """
        img_name = os.path.join(self.source_dir, self.filenames[idx])
        
        original_image = Image.open(img_name).convert('L')

        if self.transform:
            full_tensor = self.transform(original_image)  # [C, H, W]
        else:
            full_tensor = transforms.ToTensor()(original_image)  # [1, H, W]

        _, _, width = full_tensor.shape

        target_start = int(self.prediction_percentage * width)
        input_end   = int((1 - self.prediction_percentage) * width)

        input_tensor  = full_tensor[:, :, :input_end]
        target_tensor = full_tensor[:, :, target_start:]

        return input_tensor, target_tensor
    

class ImageColumnKLDivLoss(nn.Module):
    """
    KL divergence loss for comparing columns in image tensors.
    
    Calculates the KL divergence between corresponding columns of predicted
    and target images, treating each column as a probability distribution.

    :param epsilon: Small value to avoid numerical instability
    """
    def __init__(self, epsilon: float = 1e-10):
        """
        Initialize the loss function.
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate KL divergence loss between columns of two image tensors.

        :param predicted: Tensor of shape (batch_size, 1, height, width)
        :param target: Tensor of shape (batch_size, 1, height, width)
        :return: scalar loss value
        """
        if not predicted.shape == target.shape:
            raise ValueError("Predicted and target tensors must have the same shape.")
        if torch.all(predicted == target):
            raise ValueError("Predicted and target tensors are the same.")
        
        predicted = torch.clamp(predicted, min=0.0)
        target = torch.clamp(target, min=0.0)

        # add epsilon to inputs before normalization to ensure no zeros
        predicted = predicted + self.epsilon
        target = target + self.epsilon
        
        # ensure inputs are normalized
        pred_norm = predicted / predicted.sum(dim=2, keepdim=True)
        target_norm = target / target.sum(dim=2, keepdim=True)
        
        # calculate KL divergence for each column
        kl_div = F.kl_div(
            torch.log(pred_norm),  # input needs to be log-probabilities
            target_norm,      # target needs to be probabilities
            reduction='none'
        )
        
        # sum over height and width dimensions
        loss = kl_div.sum(dim=(2, 3)).mean()  # mean over batch size
        
        return loss

class CNN_Autoencoder(pl.LightningModule):
    """
    CNN-based autoencoder for image time series prediction.
    
    Implements an encoder-decoder architecture using convolutional and
    transposed convolutional layers for predicting future frames of time series data.
    """
    def __init__(self, input_chanel, chanel_list, activation_fn, batchnorm, pool_type, dropoutrate, kernel_size, padding, stride, lr):
        """
        Initialize the autoencoder.

        :param input_chanel: Number of input channels (e.g., 1 for grayscale images).
        :param chanel_list: List of output channels for each encoder layer.
        :param activation_fn: Activation function to be used in the layers.
        :param batchnorm: Boolean indicating whether to use batch normalization.
        :param pool_type: Type of pooling layer to be used ("max" or "avg").
        :param dropoutrate: Dropout rate to be used in the layers.
        :param kernel_size: Size of the convolutional kernels.
        :param padding: Padding to be used in the convolutional layers.
        :param stride: Stride to be used in the convolutional layers.
        :param lr: Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # ENCODER 
        encoder_layers = []
        in_channels = input_chanel
        for out_channels in chanel_list:
            encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm2d(out_channels))
            
            if pool_type == "max":
                encoder_layers.append(nn.MaxPool2d(kernel_size, stride, padding))
            elif pool_type == "avg":
                encoder_layers.append(nn.AvgPool2d(kernel_size, stride, padding))
            elif pool_type == "none":
                pass
            else:
                raise ValueError("pool_type must be 'max', 'avg', or 'none'.")
            if dropoutrate > 0:
                encoder_layers.append(nn.Dropout2d(dropoutrate))

            encoder_layers.append(activation_fn())
            in_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)


        # DECODER
        # We need to reverse the channels list for the decoder
        # because we are going from the bottleneck to the input.
        # The last layer of the encoder is the first layer of the decoder.
        # The first layer of the encoder is the last layer of the decoder.
        # So we need to reverse the channels list.
        decoder_layers = []
        chanel_list_rev = list(reversed(chanel_list))
        for i in range(len(chanel_list_rev) - 1):

            decoder_layers.append(nn.ConvTranspose2d(
                chanel_list_rev[i], chanel_list_rev[i+1],
                kernel_size, stride, padding
            ))

            if batchnorm:
                decoder_layers.append(nn.BatchNorm2d(chanel_list_rev[i+1]))

            if dropoutrate > 0:
                decoder_layers.append(nn.Dropout2d(dropoutrate))
            
            decoder_layers.append(activation_fn())
        # TODO: Last layer project back to original number of input channels
        decoder_layers.append(nn.ConvTranspose2d(        
            chanel_list_rev[-1], input_chanel,
            kernel_size, stride, padding
        ))
        
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, input_channels, height, width)
        :return: Reconstructed tensor of shape (batch_size, input_channels, height, width)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def training_step(self, batch):
        """
        :param batch: Tuple of input tensor and target tensor.
        :return: Loss value for the training step.
        """
        x, y = batch
        y_hat = self(x) 

        loss_instance = ImageColumnKLDivLoss()
        loss = loss_instance.forward(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)     

        return loss
    
    def validation_step(self, batch):
        """
        :param batch: Tuple of input tensor and target tensor.
        :return: Loss value for the validation step.
        """
        x, y = batch
        y_hat = self(x)
        
        loss_instance = ImageColumnKLDivLoss()
        loss = loss_instance.forward(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        """
        Congigure the optimizer for the model.
        
        :return: Adam optimizer with the specified learning rate.
        """
        return Adam(self.parameters(), lr = self.lr) 


class LossTrackerCallback(pl.Callback):
    """
    Callback to track and visualize training and validation losses.
    """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Callback triggered at the end of each validation epoch.

        :param trainer: The trainer instance.
        """
        val_loss = trainer.callback_metrics.get("val_loss")
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.train_losses.append(train_loss.cpu().item())
        if val_loss is not None:
            self.val_losses.append(val_loss.cpu().item())

    def plot_losses(self):
        """
        Plot training and validation losses over epochs.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# if __name__ == "__main__":
#     device = get_device()

#     data_dir = "data/images/harmonic/test"
#     valid_dir = "data/images/harmonic/val"
#     dataset = ImageTimeSeriesDatasetSingleFolder(data_dir, prediction_percentage=0.25)
#     validation_data = ImageTimeSeriesDatasetSingleFolder(valid_dir, prediction_percentage=0.25)

#     train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(validation_data, batch_size=32)

#     # Create specific subfolder for this model's checkpoints
#     checkpoint_dir = 'src/models/checkpoints/cnn_visual/'
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     best_model_path = None
    
#     # Check if there are saved models
#     if os.path.exists(checkpoint_dir):
#         checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
#         if checkpoint_files:
#             def get_val_loss(filename):
#                 try:
#                     val_loss_str = filename.split('val_loss=')[-1].split('-')[0].split('.ckpt')[0]
#                     return float(val_loss_str)
#                 except (ValueError, IndexError):
#                     return float('inf')  # Return infinity for files that don't match pattern
                
#             # Sort by validation loss
#             checkpoint_files.sort(key=get_val_loss)
#             best_model_path = os.path.join(checkpoint_dir, checkpoint_files[0])
#             print(f"Found best model: {best_model_path}")

#     # Ask user whether to train or load existing model
#     retrain = False
#     if best_model_path:
#         response = input("Best model found. Do you want to retrain? (y/n): ").lower()
#         retrain = response == 'y'
#     else:
#         raise FileNotFoundError("No existing model found. Retraining is required, please set retrain=True.")

#     if retrain:
#         print("Training new model...")
#         # Create model from scratch
#         model = CNN_Autoencoder(
#             input_chanel=1,
#             chanel_list=[32, 64, 128],
#             activation_fn=nn.ReLU,
#             batchnorm=True,  
#             pool_type="max", 
#             dropoutrate=0.2,
#             kernel_size=3,
#             padding=1,
#             stride=1,
#             lr=1e-3
#         )
        
#         model = model.to(device)

#         checkpoint_callback = ModelCheckpoint(
#             dirpath=checkpoint_dir,
#             filename='epoch={epoch}-val_loss={val_loss:.2f}',
#             save_top_k=3,
#             monitor='val_loss',
#             mode='min'
#         )

#         loss_tracker = LossTrackerCallback()
#         trainer = pl.Trainer(
#             max_epochs=3,
#             enable_checkpointing=True,
#             logger=False,
#             accelerator="auto",
#             callbacks=[loss_tracker, checkpoint_callback],
#         )

#         trainer.fit(model, train_loader, val_loader)
#         loss_tracker.plot_losses()
        
#         # Update best model path to the newly trained model
#         best_model_path = checkpoint_callback.best_model_path
#         print(f"New best model saved at: {best_model_path}")
#     else:
#         print(f"Loading existing model from {best_model_path}")
#         model = CNN_Autoencoder.load_from_checkpoint(best_model_path)
#         model = model.to(device)
#         model.eval()  # Set model to evaluation mode

#     # Visualize predictions with the final model
#     print("Generating predictions visualization...")
#     plot_predictions(model, dataset)
#     plot_predictions(model, validation_data)