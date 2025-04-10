import warnings
from typing import Dict, Union

import numpy as np
import pandas as pd
from pmdarima import auto_arima

from models.utils_models import check_stationarity, extract_series_from_parquet

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class AutoARIMA:
    """
    Simplified automatic ARIMA model selection and forecasting
    """
    
    def __init__(
        self,
        seasonal: bool = False,
        m: int = 1,
        trace: bool = False,
        start_p: int = 0,
        max_p: int = 5,
        start_q: int = 0, 
        max_q: int = 5,
        d: int = None,
        max_d: int = 2,
        information_criterion: str = 'aic',
        stepwise: bool = True
    ):
        """
        Initialize AutoARIMA
        
        :param seasonal: wether to use seasonal ARIMA
        :param m: seasonal period
        :param trace: whether to print progress
        :param start_p: starting value for p
        :param max_p: maximum value for p
        :param start_q: starting value for q
        :param max_q: maximum value for q
        :param d: differencing order
        :param max_d: maximum differencing order
        :param information_criterion: information criterion to use ('aic', 'bic', etc.)
        :param stepwise: whether to use stepwise search
        """
        self.seasonal = seasonal
        self.m = m
        self.trace = trace
        self.start_p = start_p
        self.max_p = max_p
        self.start_q = start_q
        self.max_q = max_q
        self.d = d
        self.max_d = max_d
        self.information_criterion = information_criterion
        self.stepwise = stepwise
        self.model = None
        self.best_params = None
        
    def fit(self, data: Union[pd.Series, np.ndarray], check_stationarity_first: bool = True) -> 'AutoARIMA':
        """
        Fit the model to the data using auto_arima

        :param data: Time series data to fit
        :return: self
        """
        if check_stationarity_first:
            is_stationary, p_value, suggested_d = check_stationarity(data)
            print(f"Series is already stationary (p-value: {p_value:.4f})")
            if not is_stationary:
                print(f"Data is not stationary (p-value: {p_value:.4f}). Suggested differencing: {suggested_d}")
                if self.d is None:
                    self.d = suggested_d
                    print(f"Setting differencing parameter d to {suggested_d}")

        self.model = auto_arima(
            data,
            start_p=self.start_p,
            max_p=self.max_p,
            start_q=self.start_q,
            max_q=self.max_q,
            d=self.d,
            max_d=self.max_d,
            seasonal=self.seasonal,
            m=self.m,
            information_criterion=self.information_criterion,
            trace=self.trace,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=self.stepwise
        )
        
        order = self.model.order
        self.best_params = {
            'p': order[0],
            'd': order[1],
            'q': order[2]
        }
        
        if self.seasonal:
            seasonal_order = self.model.seasonal_order
            self.best_params.update({
                'P': seasonal_order[0],
                'D': seasonal_order[1],
                'Q': seasonal_order[2],
                'm': seasonal_order[3]
            })
            
        return self
    
    def predict(self, n_periods: int = 20) -> np.ndarray:
        """
        Generate forecasts
        
        :param n_periods: Number of periods to forecast
        :return: Forecasted values
        """
        if self.model is None:
            raise ValueError("Model must be fit before forecasting.")
            
        return self.model.predict(n_periods=n_periods)
    
    def summary(self) -> None:
        """
        Print summary of the model and its parameters.
        """
        if self.model is None:
            raise ValueError("Model must be fit first.")
        
        print(self.model.summary())
        print(f"\nBest model parameters: {self.best_params}")
    
    def evaluate(self, test_data: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate model performance on test data, metrics are MASE and SMAPE defined as:
        MASE = mean(abs(actual - forecast) / mean(abs(actual - lagged)))
        SMAPE = mean(abs(actual - forecast) / ((abs(actual) + abs(forecast))/2))

        :param test_data: Test data to evaluate the model
        :return: Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be fit first.")
        
        preds = self.predict(n_periods=len(test_data))
        
        if not isinstance(test_data, np.ndarray):
            actual = np.asarray(test_data)
            forecast = np.asarray(preds)
        else:
            actual = test_data
            forecast = preds
        
        smape = np.mean(np.abs(actual - forecast) / ((np.abs(actual) + np.abs(forecast)) / 2 + 1e-8)) 
        
        # For MASE, we need seasonal differencing from the training data, we'll use simple one-step differencing as naive forecast
        history = self.model.arima_res_.data.orig_endog
        if len(history) > 1:
            naive_errors = np.abs(np.diff(history))
            naive_mae = np.mean(naive_errors)
            
            forecast_errors = np.abs(actual - forecast)
            
            mase = np.mean(forecast_errors) / naive_mae if naive_mae > 0 else np.inf
        else:
            print("Not enough historical data to calculate MASE, assigning NaN.")
            mase = np.nan  # not enough historical data for MASE
        
        return {
            'MASE': mase,
            'SMAPE': smape
        }
        

if __name__ == "__main__":
    data = extract_series_from_parquet('data/data_storage/harmonic_ou_parquets/test_harmonic.parquet', 35)
    print(f"Using data of length {len(data)}")

    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    print(f"Training ARIMA on {len(train_data)} samples, testing on {len(test_data)} samples")

    auto_model = AutoARIMA(seasonal=False)
    auto_model.fit(train_data, check_stationarity_first = True)
    auto_model.summary()
    
    predictions = auto_model.predict(n_periods=len(test_data))
    metrics = auto_model.evaluate(test_data)
    print(f"AutoARIMA - MASE: {metrics['MASE']:.4f}, SMAPE: {metrics['SMAPE']:.4f}")
    