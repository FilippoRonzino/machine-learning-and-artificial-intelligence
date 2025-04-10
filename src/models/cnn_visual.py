import torch
import torch.nn as nn
import torch.optim as optim
# Import the scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Import functional for the custom loss
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import traceback
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

SOURCE_IMAGE_FOLDER = '/Users/edoardoghirardo/Offline docs/github/images' # importing My TEST folder with all different pictures.
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
NUM_EPOCHS = 25
EMBEDDING_DIM = 256


# to be changed to the correct loss
class ImageColumnKLDivLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-8): #increased epsilon 
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate KL divergence loss between columns of two image tensors.
        
        :param predicted: Tensor of shape (batch_size, 1, height, width)
        :param target: Tensor of shape (batch_size, 1, height, width)
        :return: scalar loss value
        """
        # Ensure inputs are positive
        predicted = torch.clamp(predicted, min=0.0)
        target = torch.clamp(target, min=0.0)
        
        # Add epsilon and ensure numerical stability
        predicted = predicted + self.epsilon
        target = target + self.epsilon
        
        # Normalize along height dimension
        pred_sum = predicted.sum(dim=2, keepdim=True)
        target_sum = target.sum(dim=2, keepdim=True)
        
        # Avoid division by zero
        pred_norm = predicted / torch.clamp(pred_sum, min=self.epsilon)
        target_norm = target / torch.clamp(target_sum, min=self.epsilon)
        
        # Calculate KL divergence
        kl_div = F.kl_div(
            (pred_norm + self.epsilon).log(),
            target_norm,
            reduction='none'
        )
        
        # Check for NaN values and replace with zeros
        kl_div = torch.nan_to_num(kl_div, 0.0)
        
        # Sum over height and average over batch and width
        loss = kl_div.sum(dim=2).mean()
        
        return loss
    

class ImageTimeSeriesDatasetSingleFolder(Dataset):
    def __init__(self, source_dir, transform=None, overlap = 0.5):
        self.source_dir = source_dir
        self.transform = transform
        
        self.filenames = sorted([
            f for f in os.listdir(source_dir) 
            if os.path.isfile(os.path.join(source_dir, f)) 
               and f.lower().endswith('.png')
        ])
        if not self.filenames:
            raise FileNotFoundError(f"No PNG image files found in {source_dir}.")
        
        self.overlap = overlap

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.source_dir, self.filenames[idx])
        
        # load grayscale image
        original_image = Image.open(img_name).convert('L')

        # apply transform if provided, else just convert to tensor
        if self.transform:
            full_tensor = self.transform(original_image)  # [C, H, W]
        else:
            full_tensor = transforms.ToTensor()(original_image)  # [1, H, W]

        # compute width
        _, _, width = full_tensor.shape

        # get indices for the slicing
        target_start = int(0.5*(1 - self.overlap) * width)
        input_end   = int(0.5*(1 + self.overlap) * width)

        # slice input and target
        input_tensor  = full_tensor[:, :, :input_end]
        target_tensor = full_tensor[:, :, target_start:]

        return input_tensor, target_tensor



class VisualAE_variablesize(nn.Module):
    """
    A fully convolutional Autoencoder designed to handle variable input image sizes.
    It encodes the input image into a feature map and then decodes it back.
    The convolutional parameters (kernel, stride, padding) are kept consistent
    with the original fixed-size version.
    """
    def __init__(self):
        # No embedding_dim needed here as the bottleneck is a feature map
        super(VisualAE_variablesize, self).__init__()

        # --- Encoder ---
        # Takes variable size [B, 1, H, W] input
        self.encoder_conv = nn.Sequential(
            # Layer 1: [B, 1, H, W] -> [B, 64, H/2, W/2] (approx)
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Layer 2: [B, 64, H/2, W/2] -> [B, 128, H/4, W/4] (approx)
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Layer 3: [B, 128, H/4, W/4] -> [B, 512, H/8, W/8] (approx)
            nn.Conv2d(128, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
            # Output is the bottleneck feature map
        )

        # --- Decoder ---
        # Takes variable size [B, 512, H_bottleneck, W_bottleneck] input
        self.decoder_conv = nn.Sequential(
            # Layer 1: [B, 512, H/8, W/8] -> [B, 128, H/4, W/4] (approx)
            # Note: output_padding=1 helps align dimensions when stride=2 doubles the size
            nn.ConvTranspose2d(512, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Layer 2: [B, 128, H/4, W/4] -> [B, 64, H/2, W/2] (approx)
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Layer 3: [B, 64, H/2, W/2] -> [B, 1, H, W] (approx)
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid() # Output pixel values between 0 and 1
        )

    def encode(self, x):
        
        embedding_map = self.encoder_conv(x)
        return embedding_map

    def decode(self, embedding_map):
        
        reconstruction = self.decoder_conv(embedding_map)
        return reconstruction

    def forward(self, x):
        embedding_map = self.encode(x)
        reconstruction = self.decode(embedding_map)
        return reconstruction
    

if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = None  # or transforms.Compose([...]) if desired
    dataset = ImageTimeSeriesDatasetSingleFolder(SOURCE_IMAGE_FOLDER, transform=transform, overlap=0.4)

    # Use the pad_collate_fn here
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    print("DataLoader initialized.")

        # --- Initialize dataset ---
    try:
        dataset = ImageTimeSeriesDatasetSingleFolder(source_dir=SOURCE_IMAGE_FOLDER)
        print(f"Dataset initialized with {len(dataset)} images.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the SOURCE_IMAGE_FOLDER path is correct.")
        exit()
    except Exception as e:
        print(f"An error occurred during dataset initialization: {e}")
        exit()

    # --- Initialize model, optimizer, and loss function ---
    model = VisualAE_variablesize().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = ImageColumnKLDivLoss().to(device) 

    # --- Training Loop ---
    print("Starting training...")
    train_loss_list = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        num_batches = 0

        
        for batch_data in dataloader:
            # The dataloader now yields (input_batch, target_batch)
            input_batch, target_batch = batch_data

            # Skip batch if empty (can happen if all items failed in __getitem__ and were filtered)
            if input_batch.nelement() == 0 or target_batch.nelement() == 0:
                print("Skipping empty batch")
                continue

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device) # Move target if needed by loss/comparison

            # Forward pass
            recon_batch = model(input_batch) # Model reconstructs the input

            

            # Prediction Loss (Compare recon to the target_batch - last 70%)
            loss = criterion(recon_batch, target_batch) 
            # FOR FUTUR : ImageColumnKLDivLoss handles potential shape mismatches ? 

            # Backward pass and optimize
            optimizer.zero_grad()
            # Handle potential gradient issues
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
            else:
                print(f"Warning: Skipping optimizer step due to non-finite loss in Epoch {epoch+1}, Batch {num_batches+1}")


            # Accumulate loss
            if torch.isfinite(loss): # Only accumulate finite loss
                running_loss += loss.item() * input_batch.size(0)
            num_batches += 1

        # Calculate epoch loss
        if len(dataset) > 0:
            epoch_loss = running_loss / len(dataset) # Average loss per sample
            train_loss_list.append(epoch_loss)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.6f}") # Increased precision
        else:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], No data processed.")


    print("Training complete!")

    # --- Plot the training loss ---
    if train_loss_list: # Only plot if list is not empty
        plt.figure()
        plt.plot(train_loss_list)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
    else:
        print("No training loss recorded to plot.")

    # --- Visualize 5 input vs reconstructed images ---
    print("Visualizing sample reconstructions...")
    if len(dataset) > 0: # Only visualize if there's data
        model.eval()
        with torch.no_grad():
            try:
                # Grab one batch of data using an iterator
                data_iter = iter(dataloader)
                # <<< FIX: Unpack the batch correctly >>>
                input_batch, target_batch = next(data_iter)

                if input_batch.nelement() > 0:
                    # <<< FIX: Move input_batch to device >>>
                    sample_images = input_batch.to(device)

                    # Forward pass to get reconstruction
                    recon_images = model(sample_images)

                    # Show 'num_to_show' images (or fewer if batch is smaller)
                    num_to_show = min(5, sample_images.size(0))
                    input_images_to_show = sample_images[:num_to_show]
                    output_images_to_show = recon_images[:num_to_show]
                    # If you want to show the target (last 70%) as well:
                    # target_images_to_show = target_batch[:num_to_show]

                    # Concatenate input and reconstructed images for display
                    # Ensure they have the same size for make_grid - they should if recon matches input
                    combined = torch.cat([input_images_to_show, output_images_to_show], dim=0)

                    # Create a grid: num_to_show columns, 2 rows
                    grid = make_grid(combined, nrow=num_to_show)

                    plt.figure(figsize=(num_to_show * 2, 5)) # Adjust figure size
                    # Convert the grid to CPU numpy HWC format for plotting
                    plt.imshow(grid.permute(1, 2, 0).cpu().numpy().squeeze(), cmap='gray') # Added squeeze for grayscale
                    plt.title('Top: Input (First 70%) | Bottom: Reconstructed')
                    plt.axis('off')
                    plt.show()
                else:
                    print("Could not retrieve a valid batch for visualization.")

            except StopIteration:
                print("DataLoader is empty, cannot visualize.")
            except Exception as e:
                print(f"An error occurred during visualization: {e}")
                print(traceback.format_exc())
    else:
        print("Dataset is empty, skipping visualization.")
    ')