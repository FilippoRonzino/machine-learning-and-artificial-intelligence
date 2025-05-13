<div align="center">

# Visual Intuition in Time Series Forecasting: A CNN Approach Inspired by Human Perception
#### 30562 - Machine Learning and Artificial Intelligence, Bocconi University
### Edoardo Ghirardo, Francois Maurice Hoche, Giuseppe Iannone, Filippo Antonio Ronzino, Elisa Tofanelli

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
│   │   ├── checkpoints/    # Final training checkpoints for each model
│   │   ├── lightning_logs/ # Logs for PyTorch Lightning
│   │   ├── logs/           # Logs for training and evaluation status
│   │   ├── arima.py
│   │   ├── cnn_numerical.py
│   │   ├── cnn_visual.py
│   │   ├── trainer.py
│   │   ├── utils_models.py
│   │   └── checkpoints/    # Model checkpoints for experiments
│   ├── test/               # Model and pipeline tests, plus experiment checkpoints
│   │   ├── __init__.py
│   │   ├── pbt_checkpoints/
│   │   ├── pbt_results.parquet 
│   │   ├── pbt_tuning.py
│   │   ├── test_arima.py
│   │   ├── test_custom_loss.ipynb
│   │   └── test_models.ipynb
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
To replicate the core experiments, simply run the main script:

```bash
python src/models/trainer.py
```

This will launch the training pipeline as defined in the `if __name__ == "__main__":` block of `trainer.py`. The script will:
- Load model hyperparameters and configurations from `src/test/pbt_results.parquet`.
- Set up logging and training callbacks.
- Automatically train all models specified in the parameter file, using the correct data splits and early stopping settings.
- Save model checkpoints and logs for each experiment.
Note that if there are already checkpoints in the `src/models/checkpoints/` folder, the script will ask the user the option to skip the training or to retrain the models. If you choose to retrain, the script will add the new checkpoint to the existing ones.

You can customize the training process by editing the parameters in the main block of `src/models/trainer.py`, such as the number of epochs, patience, or which hyperparamters to use.
For reference on the parameter format, see `src/test/pbt_results.parquet` (generated by the pipeline in `src/test/pbt_tuning.py`).

Otherwise, you can simply evaluate pre-trained models as explained in the `/src/test/test_models.ipynb` notebook. Here, the idea is to load a model from a checkpoint and set it in evaluation mode and use helper functions to plot the predictions, losses and other metrics. 

We list here some example usages for the main functions.

- Train a model from a dataset of parameters with columns `model_type`, `dataset`, `params` (e.g. `src/test/pbt_results.parquet`):
```python
parameter_df = pd.read_parquet(parameter_df_path)

visual_min_delta = 0.1
visual_patience = 10
numerical_min_delta = 0.00001
numerical_patience = 10

patience = (visual_patience, numerical_patience)
min_delta = (visual_min_delta, numerical_min_delta)

train_models_from_dataframe(
    parameter_df=parameter_df, max_epochs=200, prediction_percentage=0.25, 
    force_retrain=False, patience=patience, min_delta=min_delta
    )
```
The function `train_models_from_dataframe()` uses `ModelTrainer` class which wraps around the two model classes `CNNTimeSeriesPredictor` and `CNN_Autoencoder` managing training, evaluation, and testing of the models. It takes as input a dataframe with the parameters for each model, the maximum number of epochs, the percentage of data to use for prediction, a boolean to force retraining and the patience and min_delta values for early stopping.
It is thought to be used in conjunction with the `pbt_tuning.py` script which generates the parameter dataframe. Ideally, one should run the `pbt_tuning.py` script first to generate the parameter dataframe and then use it to train the models, see the next example.

- Hyperparameter tuning with PBT:
```python
numerical_df = run_numerical()
visual_df = run_visual()

combined_df = pd.concat([numerical_df, visual_df], ignore_index=True)
combined_df.to_parquet("src/test/pbt_results.parquet", index=False)
```
The two functions `run_numerical` and `run_visual` will run the PBT algorithm on the numerical and visual models respectively, using the train and eval data from each data storage folder.

- Test models, generate predictions and loss plots:
```python
automated_evaluation(
        model_type=ModelType.CNN_NUMERICAL,
        dataset_type=DatasetType.ECG,
        n_plots=1,
        all_predictions=True
    )

plot_multiple_models_loss([
        (ModelType.CNN_VISUAL, DatasetType.ECG),
        (ModelType.CNN_VISUAL, DatasetType.OU),
        (ModelType.CNN_VISUAL, DatasetType.HARMONIC),
        (ModelType.CNN_VISUAL, DatasetType.SP500),
    ], include_val=False, smoothing_factor=0.6)
```
The `automated_evaluation()` function provides a streamlined way to test models and visualize their performance. It automatically locates the latest model checkpoint for the specified model and dataset combination, loads the model, and generates comprehensive visualizations including prediction plots and training/validation loss curves. This function works with both numerical time series models (showing actual vs predicted values) and visual models (displaying input, expected, and reconstructed images).  
On the other hand, `plot_multiple_models_loss()` allows you to compare the training and validation loss curves of multiple models and datasets in a single plot. By passing a list of `(ModelType, DatasetType)` pairs, it automatically locates the relevant TensorBoard event files, extracts the loss histories, and overlays the curves for easy visual comparison. It allows for including or excluding validation loss curves, and applies a smoothing factor to the curves for better readability.

- Instantiating models directly:
```python 
device = get_device()
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
    callbacks = [loss_tracker]
    )
trainer.fit(model, train_loader, val_loader)
loss_tracker.plot_losses()
```
```python 
device = get_device()
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

data_dir = "data/images/harmonic/test"
valid_dir = "data/images/harmonic/val"
dataset = ImageTimeSeriesDatasetSingleFolder(data_dir, prediction_percentage=0.25)
validation_data = ImageTimeSeriesDatasetSingleFolder(valid_dir, prediction_percentage=0.25)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(validation_data, batch_size=32)

loss_tracker = LossTrackerCallback()
trainer = pl.Trainer(
    max_epochs=3,
    enable_checkpointing=True,
    logger=False,
    accelerator="auto",
    callbacks=[loss_tracker],
)
trainer.fit(model, train_loader, val_loader)
loss_tracker.plot_losses()
```
For more customized experimentation, the `CNNTimeSeriesPredictor` and `CNN_Autoencoder` classes can also be instantiated directly. This approach allows for fine-grained control over model architecture, training parameters, and evaluation procedures seeing what's available under the automated pipeline. Indeed, the `trainer.py` script is designed to be a wrapper around the two model classes that adds checkpointing, logging and training callbacks. The two models can be trained and evaluated independently, but the pipeline is designed to work with the `ModelTrainer` class which manages the training and evaluation process for both models as explained above.

TBC