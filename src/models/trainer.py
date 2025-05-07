import os
from enum import Enum
from typing import Type, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from models.cnn_numerical import CNNTimeSeriesPredictor
from models.cnn_visual import (
    CNN_Autoencoder,
    ImageTimeSeriesDatasetSingleFolder,
    LossTrackerCallback,
)
from models.utils_models import get_device, get_val_loss, load_and_prepare_data
from visualization.utils_visualization import plot_predictions

from pytorch_lightning.loggers import TensorBoardLogger




# use: run in terminal: tensorboard --logdir lightning_logs
class LoggingCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        epoch = trainer.current_epoch
        print(f"Epoch {epoch} | val_loss: {logs['val_loss']:.4f} | train_loss: {logs['train_loss']:.4f}")
        
        # Save to CSV or other format if needed
        with open('training_log.csv', 'a') as f:
            f.write(f"{epoch},{logs['train_loss']:.6f},{logs['val_loss']:.6f}\n")

class ClassInputType(Enum):
    """Enum to specify input data type for models."""
    IMAGE = "image"
    NUMERICAL = "numerical"

class TrainingMode(Enum):
    """Enum to specify the training action."""
    RETRAIN_FROM_SCRATCH = "retrain"
    CONTINUE_TRAINING = "continue"
    USE_EXISTING = "use_existing"

class ModelTrainer:
    """
    A class to manage the training, evaluation, and testing of DL models
    for time series prediction tasks.
    """
    
    def __init__(self, model_name, model_class=None, model_params=None):
        """
        Initialize the model trainer.
        
        :param model_name: Name of the model
        :param model_class: The model class to use
        :param model_params: Parameters for model initialization
        """
        self.model_name = model_name
        self.model_class = model_class
        self.model_params = model_params or {}
        self.device = get_device()
        self.checkpoint_dir = f'src/models/checkpoints/{model_name}/'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_path = None
        self.model = None
    
    def prepare_data(self, train_dataset, test_dataset, batch_size=32):
        """
        Prepare data loaders for training and validation, shuffles training data.
        
        :param train_dataset: Training dataset
        :param val_dataset: Testing dataset
        :param batch_size: Batch size for data loaders
        :return: DataLoader for training and validation datasets
        """
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return self.train_loader, self.test_loader
    
    def find_best_model(self):
        """
        Find the best model based on validation loss.

        :return: Path to the best model checkpoint
        """
        self.best_model_path = None
        if os.path.exists(self.checkpoint_dir):
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.ckpt')]
            if checkpoint_files:                    
                checkpoint_files.sort(key=get_val_loss)
                self.best_model_path = os.path.join(self.checkpoint_dir, checkpoint_files[0])
                print(f"Found best model: {self.best_model_path}")
                # delete all other checkpoints
                for f in checkpoint_files[3:]:
                    os.remove(os.path.join(self.checkpoint_dir, f))
                print(f"Deleted other checkpoints in {self.checkpoint_dir}")
                return self.best_model_path
        return None
    
    def train(self, max_epochs=100, training_mode: TrainingMode = TrainingMode.RETRAIN_FROM_SCRATCH):
        """
        Train the model or load a pre-trained model.
        
        :param max_epochs: Maximum number of epochs for training
        :param training_mode: Action to take (RETRAIN_FROM_SCRATCH, CONTINUE_TRAINING, USE_EXISTING)
        :return: Trained model
        """
        self.find_best_model()
        model_to_train_or_load = None 

        if training_mode == TrainingMode.RETRAIN_FROM_SCRATCH:
            print("Retraining model from scratch...")
            if self.model_class is None:
                raise ValueError("Model class must be provided for training")
            model_to_train_or_load = self.model_class(**self.model_params)
            model_to_train_or_load = model_to_train_or_load.to(self.device)
            # Reset best_model_path as we are starting fresh
            self.best_model_path = None
            last_epoch = -1


        elif training_mode == TrainingMode.CONTINUE_TRAINING:
            if self.best_model_path:
                print(f"Continuing training from existing model: {self.best_model_path}")
                model_to_train_or_load = self.model_class.load_from_checkpoint(self.best_model_path)
                model_to_train_or_load = model_to_train_or_load.to(self.device)
                # Extract the last epoch number from checkpoint filename
                try:
                    last_epoch = int(os.path.basename(self.best_model_path).split('epoch=')[2].split('-')[0])
                    print(f"Resuming from epoch {last_epoch}")
                except (IndexError, ValueError):
                    print("Could not determine last epoch from filename")
                    last_epoch = -1
            else:
                print("No existing model found to continue. Training new model from scratch...")
                if self.model_class is None:
                    raise ValueError("Model class must be provided for training")
                model_to_train_or_load = self.model_class(**self.model_params)
                model_to_train_or_load = model_to_train_or_load.to(self.device)
                training_mode = TrainingMode.RETRAIN_FROM_SCRATCH

        elif training_mode == TrainingMode.USE_EXISTING:
            if self.best_model_path:
                print(f"Loading existing model from {self.best_model_path}")
                model_to_train_or_load = self.model_class.load_from_checkpoint(self.best_model_path)
                model_to_train_or_load = model_to_train_or_load.to(self.device)
                model_to_train_or_load.eval()
                self.model = model_to_train_or_load 
                return self.model
            else:
                raise FileNotFoundError("Tried to load existing model, but no checkpoint found.")
        
        if training_mode == TrainingMode.RETRAIN_FROM_SCRATCH or training_mode == TrainingMode.CONTINUE_TRAINING:
            if model_to_train_or_load is None:
                raise RuntimeError("Model was not initialized for training.")

            self.model = model_to_train_or_load

            # Create the checkpoint callback as usual
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.checkpoint_dir,
                filename='epoch={epoch}-val_loss={val_loss:.6f}',
                save_top_k=3,
                monitor='val_loss',
                mode='min',
            )

            # Create an early stopping callback
            early_stopping_callback = EarlyStopping(
                monitor='val_loss',
                patience=10,
                min_delta=0.1,
                mode='min',
                verbose=True
            )

            loss_tracker = LossTrackerCallback()

            logger = TensorBoardLogger("lightning_logs", name=self.model_name)


            trainer = pl.Trainer(
                max_epochs=max_epochs + last_epoch,
                enable_checkpointing=True,
                logger=logger, 
                accelerator="auto", 
                callbacks=[loss_tracker, checkpoint_callback, early_stopping_callback],
            )
            # Set the current epoch to the last epoch
            trainer.fit_loop.epoch_progress.current.increment_by(last_epoch+1)

            print(f"Starting trainer.fit for {max_epochs} epochs...")

            trainer.fit(self.model, self.train_loader, self.test_loader, ckpt_path=self.best_model_path)
            loss_tracker.plot_losses()

            if hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
                self.best_model_path = checkpoint_callback.best_model_path
                print(f"Training finished. Best model saved at: {self.best_model_path}")
            else:
                print("Training finished, but couldn't find best_model_path from callback.")
                self.find_best_model()


        elif self.model is None:
            raise RuntimeError("Model training/loading process failed.")

        self.model.eval() 
        return self.model

    
    def evaluate(self, test_datasets, plot_function=plot_predictions):
        """
        Evaluate the model on test datasets and visualize predictions.
        
        :param test_datasets: List of datasets to evaluate
        :param plot_function: Function to visualize predictions
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() first.")
            
        print("Generating predictions visualization...")
        for dataset in test_datasets:
            plot_function(self.model, dataset)
    
    def ask_retrain(self) -> TrainingMode:
        """
        Ask user whether to retrain, continue training, or use an existing model.

        :return: TrainingMode enum indicating user's choice
        """
        if self.find_best_model(): 
            while True: 
                response = input(
                    f"Best model found for {self.model_name} ({os.path.basename(self.best_model_path)}).\n"
                    "Options: (r)etrain from scratch, (c)ontinue training, (u)se existing? [u]: "
                ).lower().strip()

                if response == 'r':
                    return TrainingMode.RETRAIN_FROM_SCRATCH
                elif response == 'c':
                    return TrainingMode.CONTINUE_TRAINING
                elif response == 'u' or response == '': 
                    return TrainingMode.USE_EXISTING
                else:
                    print("Invalid input. Please enter 'r', 'c', or 'u'.")
        else:
            print("No existing model found. Will train new model from scratch.")
            return TrainingMode.RETRAIN_FROM_SCRATCH

def train_and_evaluate_model(
    model_name: str,
    model_class: Type,
    model_params: dict,
    input_type: ClassInputType,
    data_path: Union[str, dict],
    prediction_percentage: float = 0.25,
    batch_size: int = 32,
    max_epochs: int = 100,
    force_retrain: bool = False,
    n_plots: int = 5
):
    """
    Utility function to train and evaluate a model on specified datasets.
    
    :param model_name: Name of the model
    :param model_class: Class of the model to be trained, e.g. CNNTimeSeriesPredictor
    :param model_params: Parameters for the model
    :param input_type: Type of input data (image or numerical)
    :param data_path: Path to the dataset or a dictionary with train/test paths
    :param prediction_percentage: Percentage of data to be used for prediction
    :param batch_size: Batch size for training
    :param max_epochs: Maximum number of epochs for training
    :param force_retrain: Whether to force retraining of the model
    :param n_plots: Number of plots to generate for evaluation
    :return: Trained model
    """

    if input_type == ClassInputType.IMAGE:
        if not isinstance(data_path, dict) or 'train' not in data_path or 'test' not in data_path:
            raise ValueError("For image data, data_path must be a dictionary with 'train' and 'test' keys")
        
        train_dataset = ImageTimeSeriesDatasetSingleFolder(data_path['train'], prediction_percentage=prediction_percentage)
        val_dataset = ImageTimeSeriesDatasetSingleFolder(data_path['test'], prediction_percentage=prediction_percentage)
        test_datasets = [val_dataset]
        
    elif input_type == ClassInputType.NUMERICAL:
        if isinstance(data_path, dict):
            file_path_train = data_path.get('train')
            file_path_test = data_path.get('test')
        else:
            file_path_train = os.path.join(data_path, "train_harmonic.parquet")
            file_path_test = os.path.join(data_path, "test_harmonic.parquet")
        
        print("Loading and preparing numerical data...")
        X_train, y_train = load_and_prepare_data(file_path_train, prediction_percentage=prediction_percentage)
        X_test, y_test = load_and_prepare_data(file_path_test, prediction_percentage=prediction_percentage)
        
        print("X_train sample shape:", X_train[0].shape)
        print("y_train sample shape:", y_train[0].shape)
        print(len(X_train), "samples loaded for training")
        print(len(X_test), "samples loaded for testing")
        
        train_dataset = torch.utils.data.TensorDataset(torch.stack(X_train), torch.stack(y_train))
        val_dataset = torch.utils.data.TensorDataset(torch.stack(X_test), torch.stack(y_test))
        test_datasets = [val_dataset]
    
    else:
        raise ValueError(f"Unsupported input type: {input_type}")
    
    trainer = ModelTrainer(model_name, model_class, model_params)
    trainer.prepare_data(train_dataset, val_dataset, batch_size)
    
    retrain = force_retrain or trainer.ask_retrain()
    model = trainer.train(max_epochs, retrain)
    
    if test_datasets:
        if not isinstance(test_datasets, list):
            test_datasets = [test_datasets]
        
        for dataset in test_datasets:
            plot_predictions(model, dataset, n_plots=n_plots)
        
    return model


if __name__ == "__main__":
    

    # Example for CNN_Visual model
    print("====== CNN Visual Model Example ======")
    # Define model parameters for visual model
    cnn_visual_params = {
        "input_chanel": 1,
        "chanel_list": [32, 64],
        "activation_fn": torch.nn.ReLU,
        "batchnorm": True,
        "pool_type": "max",
        "dropoutrate": 0.2,
        "kernel_size": 5,
        "padding": 2,
        "stride": 1,
        "lr": 1e-3
    }
    
    # Define data paths for image data
    image_data_paths = {
        "train": "data/images/harmonic/train",
        "test": "data/images/harmonic/val"
    }
    
    # Train and evaluate image model
    visual_model = train_and_evaluate_model(
        model_name="cnn_visual",
        model_class=CNN_Autoencoder,
        model_params=cnn_visual_params,
        input_type=ClassInputType.IMAGE,
        data_path=image_data_paths,
        prediction_percentage=0.25,
        max_epochs=100
    )
    
    # Example for CNN_Numerical model
    print("\n====== CNN Numerical Model Example ======")
    
    # Define model parameters for numerical model
    cnn_numerical_params = {
        "input_features": 1,
        "input_seq_len": 60,
        "output_features": 1,
        "model_output_seq_len": 60,
        "actual_prediction_len": 20,
        "cnn_layers": 3,
        "kernel_size": 3,
        "base_filters": 32,
        "fc_size": 128,
        "lr": 1e-3
    }
    
    # Define data paths for numerical data
    numerical_data_paths = {
        "train": "data/data_storage/harmonic_ou_parquets/train_harmonic.parquet",
        "test": "data/data_storage/harmonic_ou_parquets/val_harmonic.parquet"
    }
    
    # Train and evaluate numerical model
    numerical_model = train_and_evaluate_model(
        model_name="cnn_numerical",
        model_class=CNNTimeSeriesPredictor,
        model_params=cnn_numerical_params,
        input_type=ClassInputType.NUMERICAL,
        data_path=numerical_data_paths,
        prediction_percentage=0.25,
        max_epochs=100,
        batch_size=32
    )
    
    print("Both models have been successfully trained and evaluated.")