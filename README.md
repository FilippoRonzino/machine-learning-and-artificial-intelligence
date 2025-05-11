<div align="center">

# Visual Intuition in Time Series Forecasting: A CNN Approach Inspired by Human Perception
### 30562 - Machine Learning and Artificial Intelligence, Bocconi University
## Authors: Edoardo Ghirardo, Francois Maurice Hoche, Giuseppe Iannone, Filippo Antonio Ronzino, Elisa Tofanelli

</div>

## Abstract
This project investigates the potential of Convolutional Neural Networks (CNNs) to forecast time series data by emulating human visual intuition. Humans often demonstrate superior pattern recognition and extrapolation abilities when time series are presented graphically rather than as raw numerical data. Building on this observation, this research explores whether CNNs, traditionally used for image processing, can achieve a similar level of "visual intuition" for temporal data. The study evaluates CNN performance in two primary areas: the numerical continuation of a time series and its visual (graphical) completion. The project aims to determine if machines can effectively "see" future trends in sequences, mirroring human perceptual capabilities. The methodology involves training CNNs on diverse datasets, including synthetic harmonic and Ornstein-Uhlenbeck process data, as well as real-world ECG and S&P 500 pricing data, presented in both numerical and visual formats.

## Structure
**Repo organization:**
```
.
├── README.md               # Project documentation
├── LICENSE                 # License information
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup script
├── __init__.py
├── .gitignore
├── data/                   # Data handling and storage
│   ├── __init__.py
│   ├── plot_images_pipeline.py
│   ├── data_fetchers/      # Scripts to fetch and preprocess datasets
│   │   ├── data_ecg_fetcher.py
│   │   ├── data_sp500_fetcher.py
│   │   └── data_synthetic_fetcher.py
│   ├── data_storage/       # Parquet storage for datasets (train/val/test splits)
│   │   ├── ecg_parquets/
│   │   ├── harmonic_ou_parquets/
│   │   └── sp500_parquets/
│   └── images/             # Generated images for each dataset (train/val/test splits)
│       ├── ecg/
│       ├── harmonic/
│       ├── ou/
│       └── sp500/
├── data_analysis/          # Notebooks and utilities for data exploration
│   ├── compare_data_complexity.ipynb
│   ├── data_analysis-ECG_data.ipynb
│   ├── data_analysis-OU_data.ipynb
│   ├── data_analysis-financial_data.ipynb
│   ├── data_analysis-harmonic_data.ipynb
│   └── data_analysis_utils.py
├── src/                    # Source code for models, training, and utilities
│   ├── __init__.py
│   ├── loss.ipynb
│   ├── models/
│   │   ├── __init__.py
│   │   ├── arima.py
│   │   ├── cnn_numerical.py
│   │   ├── cnn_visual.py
│   │   ├── trainer.py
│   │   ├── utils_models.py
│   │   └── checkpoints/    # Model checkpoints for experiments
│   ├── test/               # Model and pipeline tests, plus experiment checkpoints
│   │   ├── __init__.py
│   │   ├── pbt_checkpoints/
│   │   ├── pbt_tuning.py
│   │   ├── test_arima.py
│   │   └── test_models_visual.ipynb
│   ├── utils.py
│   └── visualization/
│       ├── __init__.py
│       └── utils_visualization.py
```

**Main folders and files:**
- `data/`: Data fetching, storage, and image generation. In particular `data/data_fetchers/` contains:
    - `data_ecg_fetcher.py`: ECG data fetching and preprocessing from MIT-BIH Normal Sinus Rhythm Database;
    - `data_sp500_fetcher.py`: S&P 500 financial data fetching and preprocessing from Yahoo! Finance API;
    - `data_synthetic_fetcher.py`: Synthetic data generation and preprocessing as mathematically defined in the report.
- `data_analysis/`: Preliminary data exploration and analysis notebooks.
- `src/`: Model definitions, training scripts, utilities, and tests. It contains the main files that are used to train and evaluate the models. In particular:
    - `src/models/`: Contains the model definitions, including CNN architectures and training scripts:
        - `cnn_numerical.py`: CNN model for numerical data, main class is `CNNTimeSeriesPredictor`;
        - `cnn_visual.py`: CNN model for visual data, main class is `CNN_Autoencoder`;
        - `trainer.py`: Training and evaluation scripts with the main class `ModelTrainer` which takes as arguments the model name, type and its parameters;
        - `utils_models.py`: Utility functions for models, time series handling and data preparation.
    - `src/test/`: Contains tests for the models and pipelines, as well as experiment checkpoints. The core file is `pbt_tuning.py` which implements the Population Based Training (PBT) algorithm for hyperparameter tuning.
    - `src/visualization/`: Contains utilities for visualizing the results of the models.


## Installation
To install the package, clone the repository and, inside it, run:
```bash
pip install .
```
This will install our `ai_project` package and its dependencies listed in `requirements.txt`.


## Usage
To replicate the experiments, simply run the main script:

```bash
python src/models/trainer.py
```

This will launch the training pipeline as defined in the `if __name__ == "__main__":` block of `trainer.py`. The script will:
- Load model hyperparameters and configurations from `src/test/pbt_results.parquet`.
- Set up logging and training callbacks.
- Automatically train all models specified in the parameter file, using the correct data splits and early stopping settings.
- Save model checkpoints and logs for each experiment.

You can customize the training process by editing the parameters in the main block of `src/models/trainer.py`, such as the number of epochs, patience, or which models to retrain.  
For reference on the parameter format, see `src/test/pbt_results.parquet` (generated by the pipeline in `src/test/pbt_tuning.py`).

Otherwise, you can simply evaluate pre-trained models as explained in the `/src/test/test_models_visual.ipynb` notebook. Here, the idea is to load a model from a checkpoint and set it in evaluation mode and use helper functions to plot the predictions, losses and other metrics. For example, to load a pre-trained `CNNTimeSeriesPredictor` model for ECG data, you can use the following code snippet:

```python
base_dir = os.getcwd()
ckpt_file = "epoch=epoch=14-val_loss=val_loss=0.014259.ckpt"
model_ecg_numerical = CNNTimeSeriesPredictor.load_from_checkpoint(
    os.path.join(base_dir, "checkpoints/cnn_numerical_ecg", ckpt_file),
    map_location=torch.device("cpu")
)
model_ecg_numerical.eval()
file_path_test = os.path.join(base_dir, "..", "..", "data", "data_storage", "ecg_parquets", "test_ecg.parquet")
X_test, y_test = load_and_prepare_data(file_path_test, prediction_percentage=0.25)
val_dataset_ecg = TensorDataset(torch.stack(X_test), torch.stack(y_test))
plot_predictions_numerical_model(model_ecg_numerical, val_dataset_ecg, n_plots=1, all_predictions = True)
```

Single main pipelines instead can be tested using the `if __name__ == "__main__":` sections modified as needed, as an example see the training of a `CNNTimeSeriesPredictor` instance on syntetic harmonic data:

```python
model = CNNTimeSeriesPredictor(
        input_features = 1,
        input_seq_len = 60,
        output_features = 1,
        model_output_seq_len = 60,  
        actual_prediction_len = 20,     
        cnn_layers = 3,
        kernel_size = 3,
        base_filters = 32,
        fc_size = 128,
        lr = 0.001
    )

model = model.to(device)

file_path_train = "data/data_storage/harmonic_ou_parquets/train_harmonic.parquet"
file_path_test = "data/data_storage/harmonic_ou_parquets/test_harmonic.parquet"
X_train, y_train = load_and_prepare_data(file_path_train, prediction_percentage=0.25)
X_test, y_test = load_and_prepare_data(file_path_test, prediction_percentage=0.25)
train_dataset = TensorDataset(torch.stack(X_train), torch.stack(y_train))
val_dataset = TensorDataset(torch.stack(X_test), torch.stack(y_test))
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True, num_workers=7)
val_loader = DataLoader(val_dataset, batch_size = 32, num_workers=7)

loss_tracker = LossTrackerCallback()
trainer = pl.Trainer(
    max_epochs = 1,
    enable_checkpointing = False,
    logger = False,
    accelerator = "auto",
    callbacks = [loss_tracker],
)

trainer.fit(model, train_loader, val_loader)
loss_tracker.plot_losses()
plot_predictions(model, val_dataset, n_plots=5)
```

All other single pipelines can be run in a similar way, by modifying the `if __name__ == "__main__":` block of the corresponding script.