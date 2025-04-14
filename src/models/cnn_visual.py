import os
import random

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


class ImageTimeSeriesDatasetSingleFolder(Dataset):
    def __init__(self, source_dir, transform=None, prediction_percentage = 0.25):
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
        return len(self.filenames)

    def __getitem__(self, idx):
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
    def __init__(self, epsilon: float = 1e-10):
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
    def __init__(self, input_chanel, chanel_list, activation_fn, batchnorm, pool_type, dropoutrate, kernel_size, padding, stride, lr):
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
            # if dropoutrate > 0:
                # encoder_layers.append(nn.Dropout2d(dropoutrate))

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

            # if dropoutrate > 0:
                # decoder_layers.append(nn.Dropout2d(dropoutrate))
            
            decoder_layers.append(activation_fn())
        # TODO: Last layer project back to original number of input channels
        decoder_layers.append(nn.ConvTranspose2d(        
            chanel_list_rev[-1], input_chanel,
            kernel_size, stride, padding
        ))
        
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self(x) 

        loss_instance = ImageColumnKLDivLoss()
        loss = loss_instance.forward(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)     

        return loss
    
    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        
        loss_instance = ImageColumnKLDivLoss()
        loss = loss_instance.forward(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr = self.lr) 


class LossTrackerCallback(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.train_losses.append(train_loss.cpu().item())
        if val_loss is not None:
            self.val_losses.append(val_loss.cpu().item())

    def plot_losses(self):
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


def visualize_predictions(model, dataset, n_images=5):
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


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")


    data_dir = "data/images/harmonic/test"
    valid_dir = "data/images/harmonic/val"
    dataset = ImageTimeSeriesDatasetSingleFolder(data_dir, prediction_percentage=0.25)
    validation_data = ImageTimeSeriesDatasetSingleFolder(valid_dir, prediction_percentage=0.25)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=32)

    model = CNN_Autoencoder(
        input_chanel=1,
        chanel_list=[32, 64, 128],
        activation_fn=nn.ReLU,
        batchnorm=True,  
        pool_type="max", 
        dropoutrate=0.2,
        kernel_size=3,
        padding=1,
        stride=1,
        lr=1e-3
    )

    model = model.to(device)

    loss_tracker = LossTrackerCallback()
    trainer = pl.Trainer(
        max_epochs=15,
        enable_checkpointing=False,
        logger=False,
        accelerator="auto",
        callbacks=[loss_tracker]
    )

    trainer.fit(model, train_loader, val_loader)

    loss_tracker.plot_losses()

    visualize_predictions(model, dataset)
    visualize_predictions(model, validation_data)

