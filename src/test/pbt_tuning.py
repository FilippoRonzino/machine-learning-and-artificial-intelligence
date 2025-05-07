import os
from collections import OrderedDict

import pandas as pd
import pytorch_lightning as pl
import ray
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search.sample import Domain

from models.cnn_numerical import CNNTimeSeriesPredictor
from models.cnn_visual import CNN_Autoencoder, ImageTimeSeriesDatasetSingleFolder
from models.trainer import ClassInputType
from models.utils_models import load_and_prepare_data

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class PBTHyperparameterTuning:
    """
    Class for performing Population-Based Training (PBT) hyperparameter tuning.
    """
    def __init__(self, model_name, model_class, input_type, data_path, num_samples=5, max_epochs=10, gpus_per_trial=0.5, prediction_percentage=0.25):
        self.model_name = model_name
        self.model_class = model_class
        self.input_type = input_type
        self.data_path = data_path
        self.num_samples = num_samples
        self.max_epochs = max_epochs
        self.prediction_percentage = prediction_percentage
        self.gpus_per_trial = gpus_per_trial if torch.cuda.is_available() else 0
        if gpus_per_trial > 0 and not torch.cuda.is_available():
            print("WARNING: GPU requested but not available. Using CPU instead.")
        self.checkpoint_dir = os.path.abspath(f'src/test/pbt_checkpoints/{model_name}/')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.param_space = self._get_param_space()

    def _get_param_space(self):
        """
        Define the hyperparameter search space for PBT tuning depending on the model class.

        :return: dict
        """
        if self.model_class == CNN_Autoencoder:
            return {
                "input_chanel": 1,
                "chanel_list": [128, 256, 512],
                "activation_fn": tune.choice(["relu", "leaky_relu", "elu"]),
                "batchnorm": tune.choice([True, False]),
                "pool_type": tune.choice(["max", "avg", "none"]),
                "dropoutrate": tune.uniform(0.0, 0.5),
                "kernel_size": tune.choice([3, 5, 7]),
                "padding": tune.sample_from(lambda spec: (spec.config["kernel_size"] - 1) // 2),
                "stride": 1,
                "lr": 1e-3,
                "batch_size": tune.choice([16, 32])
            }
        elif self.model_class == CNNTimeSeriesPredictor:
            return {
                "input_features": 1,
                "input_seq_len": 60,
                "output_features": 1,
                "model_output_seq_len": 60,
                "actual_prediction_len": 20,
                "cnn_layers": tune.choice([2, 3, 4, 5]),
                "kernel_size": tune.choice([3, 5, 7]),
                "base_filters": tune.choice([16, 32, 64]),
                "fc_size": tune.choice([64, 128, 256]),
                "lr": 1e-3,
                "batch_size": tune.choice([16, 32])
            }
        else:
            raise ValueError("Unsupported model class for PBT tuning, please use CNN_Autoencoder or CNNTimeSeriesPredictor.")
        
    def _prepare_dataset(self):
        """
        Prepare the dataset for training and validation as in trainer.py.

        :return: Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]
        """
        if self.input_type == ClassInputType.IMAGE:
            train = ImageTimeSeriesDatasetSingleFolder(self.data_path['train'], prediction_percentage=self.prediction_percentage)
            val = ImageTimeSeriesDatasetSingleFolder(self.data_path['val'], prediction_percentage=self.prediction_percentage)
        else:
            X_train, y_train = load_and_prepare_data(self.data_path['train'], prediction_percentage=self.prediction_percentage)
            X_val, y_val = load_and_prepare_data(self.data_path['val'], prediction_percentage=self.prediction_percentage)
            train = torch.utils.data.TensorDataset(torch.stack(X_train), torch.stack(y_train))
            val = torch.utils.data.TensorDataset(torch.stack(X_val), torch.stack(y_val))
        return train, val

    def _train_model(self, config):
        """
        Training function for the model. This function is called by Ray Tune.

        :param config: dict, hyperparameters for the model
        """
        train_ds, val_ds = self._prepare_dataset()
        
        # Create a copy of config to avoid modifying the original
        model_config = config.copy()
        batch_size = model_config.pop("batch_size", 32)
        
        if self.model_class == CNN_Autoencoder and "activation_fn" in model_config:
            if model_config["activation_fn"] == "relu":
                model_config["activation_fn"] = torch.nn.ReLU
            elif model_config["activation_fn"] == "leaky_relu":
                model_config["activation_fn"] = torch.nn.LeakyReLU
            elif model_config["activation_fn"] == "elu":
                model_config["activation_fn"] = torch.nn.ELU
        
        model = self.model_class(**model_config)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
        
        callbacks = [
            TuneReportCallback({"val_loss": "val_loss"}, on="validation_end"),
            EarlyStopping(monitor="val_loss", patience=3, mode="min")
        ]
        
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            enable_checkpointing=False,
            callbacks=callbacks,
            logger=False,
            accelerator="auto",
            deterministic=True,
        )
        
        trainer.fit(model, train_loader, val_loader)

    def run_pbt(self):
        """
        Run the PBT tuning process.

        :return: dict, best hyperparameters found during tuning
        """
        ray.init(ignore_reinit_error=True)
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="val_loss",
            mode="min",
            perturbation_interval=3,
            hyperparam_mutations={k: v for k, v in self.param_space.items() if isinstance(v, Domain)},
        )
        analysis = tune.run(
            self._train_model,
            config=self.param_space,
            scheduler=scheduler,
            num_samples=self.num_samples,
            resources_per_trial={"cpu": 1, "gpu": self.gpus_per_trial},
            storage_path=self.checkpoint_dir,
            name=f"{self.model_name}_pbt",
            verbose=1,
        )
        ray.shutdown()
        best_config = analysis.get_best_config(metric="val_loss", mode="min")
        return best_config


def run_visual():
    """
    Run PBT hyperparameter tuning for visual CNN models on different image datasets.
    
    Returns:
        DataFrame with columns: model_type, dataset, params
    """
    image_datasets = OrderedDict([
        ("ecg", {
            "train": os.path.join(PROJECT_ROOT, "data/images/ecg/train"),
            "val": os.path.join(PROJECT_ROOT, "data/images/ecg/val")
        }),
        ("harmonic", {
            "train": os.path.join(PROJECT_ROOT, "data/images/harmonic/train"),
            "val": os.path.join(PROJECT_ROOT, "data/images/harmonic/val")
        }),
        ("ou", {
            "train": os.path.join(PROJECT_ROOT, "data/images/ou/train"),
            "val": os.path.join(PROJECT_ROOT, "data/images/ou/val")
        }),
        ("sp500", {
            "train": os.path.join(PROJECT_ROOT, "data/images/sp500/train"),
            "val": os.path.join(PROJECT_ROOT, "data/images/sp500/val")
        })
    ])
    
    results = []
    
    for dataset_name, paths in image_datasets.items():
        if not os.path.exists(paths["train"]) or not os.path.exists(paths["val"]):
            raise FileNotFoundError(f"Dataset {dataset_name} not found in specified paths.")
        print(f"Running PBT tuning on {dataset_name} visual dataset...")
        
        pbt = PBTHyperparameterTuning(
            model_name=f"cnn_visual_{dataset_name}_pbt",
            model_class=CNN_Autoencoder,
            input_type=ClassInputType.IMAGE,
            data_path=paths,
            num_samples=3,
            max_epochs=15,
            gpus_per_trial=1,
            prediction_percentage=0.25
        )
        
        try:
            best_config = pbt.run_pbt()
            print(f"Best config for {dataset_name} visual dataset:", best_config)
            
            results.append({
                "model_type": "cnn_v",
                "dataset": dataset_name,
                "params": best_config
            })
        except Exception as e:
            print(f"Error tuning {dataset_name} visual dataset: {str(e)}")
    
    return pd.DataFrame(results)

def run_numerical():
    """
    Run PBT hyperparameter tuning for numerical CNN models on different time series datasets.
    
    Returns:
        DataFrame with columns: model_type, dataset, params
    """
    numerical_datasets = OrderedDict([
        ("ecg", {
            "train": os.path.join(PROJECT_ROOT, "data/data_storage/ecg_parquets/train_ecg.parquet"),
            "val": os.path.join(PROJECT_ROOT, "data/data_storage/ecg_parquets/val_ecg.parquet")
        }),
        ("harmonic_ou", {
            "train": os.path.join(PROJECT_ROOT, "data/data_storage/harmonic_ou_parquets/train_harmonic.parquet"),
            "val": os.path.join(PROJECT_ROOT, "data/data_storage/harmonic_ou_parquets/val_harmonic.parquet")
        }),
        ("sp500", {
            "train": os.path.join(PROJECT_ROOT, "data/data_storage/sp500_parquets/train_sp500.parquet"),
            "val": os.path.join(PROJECT_ROOT, "data/data_storage/sp500_parquets/val_sp500.parquet")
        })
    ])
    
    results = []
    
    for dataset_name, paths in numerical_datasets.items():
        if not os.path.exists(paths["train"]) or not os.path.exists(paths["val"]):
            raise FileNotFoundError(f"Dataset {dataset_name} not found in specified paths.")
        print(f"Running PBT tuning on {dataset_name} numerical dataset...")
        
        pbt = PBTHyperparameterTuning(
            model_name=f"cnn_numerical_{dataset_name}_pbt",
            model_class=CNNTimeSeriesPredictor,
            input_type=ClassInputType.NUMERICAL,
            data_path=paths,
            num_samples=3,
            max_epochs=15,
            gpus_per_trial=1,
            prediction_percentage=0.25
        )
        
        try:
            best_config = pbt.run_pbt()
            print(f"Best config for {dataset_name} numerical dataset:", best_config)
            
            results.append({
                "model_type": "cnn_n",
                "dataset": dataset_name,
                "params": best_config
            })
        except Exception as e:
            print(f"Error tuning {dataset_name} numerical dataset: {str(e)}")

    return pd.DataFrame(results)



if __name__ == "__main__":
    print("Running PBT tuning for numerical datasets...")
    numerical_df = run_numerical()
    
    print("\nRunning PBT tuning for visual datasets...")
    visual_df = run_visual()
    
    combined_df = pd.concat([numerical_df, visual_df], ignore_index=True)
    
    combined_df.to_parquet(os.path.join(PROJECT_ROOT, "src/test/pbt_results.parquet"), index=False)
    print(f"Results saved to {os.path.join(PROJECT_ROOT, 'src/test/pbt_results.parquet')}")