import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from models.cnn_numerical.cnn_numerical_utils import load_and_prepare_data
from visualization.utils_visualization import plot_actual_vs_predicted
import os

class CNNModel:
    def __init__(self, input_shape, output_steps, learning_rate=0.001):
        self.input_shape = input_shape
        self.output_steps = output_steps
        self.learning_rate = learning_rate
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=self.input_shape),
            Conv1D(64, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(100, activation='relu'),
            Dense(self.output_steps)
        ])
        model.compile(optimizer=Adam(self.learning_rate), loss='mse')
        return model

    def forward(self, x):
        return self.model.predict(x)

    def train(self, X_train, y_train, epochs=30, batch_size=32, validation_split=0.2,
              model_path="best_model.h5", patience=5):
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
        
        history = self.model.fit(X_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_split=validation_split,
                                 callbacks=[checkpoint, early_stopping])
        return history

    def evaluate(self, X_test, y_test, n_plot=0):
        loss = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss}")
        
        for i in range(n_plot):
            sample_input = X_test[i].reshape(1, *self.input_shape)
            y_pred_full = self.forward(sample_input)  
            y_pred_future = y_pred_full[0][-20:]
            x_part = X_test[i].flatten()
            y_true_future = y_test[i][-20:]
            y_true_combined = np.concatenate((x_part, y_true_future))
            y_pred_combined = np.concatenate((x_part, y_pred_future))
            plot_actual_vs_predicted(y_true_combined, y_pred_combined, 25)

        return loss

    def summary(self):
        self.model.summary()

if __name__ == "__main__":
    
    cnn_model = CNNModel(input_shape=(60, 1), output_steps=60)
    cnn_model.summary()
    
    base_dir = "/Users/giuseppeiannone/machine-learning-and-artificial-intelligence"
    file_path_train = os.path.join(base_dir, "data", "data_storage", "harmonic_ou_parquets", "train_harmonic.parquet")
    file_path_test = os.path.join(base_dir, "data", "data_storage", "harmonic_ou_parquets", "test_harmonic.parquet")

    X_train, y_train = load_and_prepare_data(file_path_train, prediction_percentage=25)
    X_test, y_test = load_and_prepare_data(file_path_test, prediction_percentage=25)
    
    print("X shape:", X_train[0].shape)
    print("y shape:", y_train[0].shape)
    print(len(X_train), "samples loaded for training")
    print(len(X_test), "samples loaded for testing")

    # Train 
    history = cnn_model.train(X_train, y_train, epochs=10, batch_size=32,
                              validation_split=0.2, model_path="best_model.h5", patience=5)
    
    print("Model training complete and best model saved.")

    # Evaluate with plots
    cnn_model.evaluate(X_test, y_test, n_plot=2)


