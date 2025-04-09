<<<<<<< HEAD
from data.data_loader import load_time_series_parquet, check_stationarity
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Optional, Tuple, Dict, Any, Union


class AutoARIMA:
    """
    Class for automatic ARIMA model selection and fitting using pmdarima's auto_arima
    """
    
    def __init__(
        self,
        start_p: int = 0,
        max_p: int = 5,
        start_q: int = 0,
        max_q: int = 5,
        d: Optional[int] = None,
        start_P: int = 0,
        max_P: int = 2,
        start_Q: int = 0,
        max_Q: int = 2,
        max_D: int = 1,
        seasonal: bool = True,
        m: int = 1,  # Default non-seasonal
        information_criterion: str = 'aic',
        trace: bool = False,
        error_action: str = 'warn',
        stepwise: bool = True
    ):
        """
        Initialize AutoARIMA with parameters for model selection
        
        Parameters:
        -----------
        start_p, max_p : int
            Lower and upper bounds for p (AR order)
        start_q, max_q : int
            Lower and upper bounds for q (MA order)
        d : Optional[int]
            Order of differencing. If None, it will be determined automatically
        start_P, max_P : int
            Lower and upper bounds for seasonal P
        start_Q, max_Q : int
            Lower and upper bounds for seasonal Q
        max_D : int
            Maximum seasonal differencing order
        seasonal : bool
            Whether to fit seasonal components
        m : int
            Seasonal periodicity (e.g., m=12 for monthly data with yearly seasonality)
        information_criterion : str
            Information criterion to use for model selection ('aic', 'bic', etc.)
        trace : bool
            Whether to print status during fitting
        error_action : str
            How to handle errors in model fitting
        stepwise : bool
            Whether to use stepwise search vs. grid search
        """
        self.start_p = start_p
        self.max_p = max_p
        self.start_q = start_q
        self.max_q = max_q
        self.d = d
        self.start_P = start_P
        self.max_P = max_P
        self.start_Q = start_Q
        self.max_Q = max_Q
        self.max_D = max_D
        self.seasonal = seasonal
        self.m = m
        self.information_criterion = information_criterion
        self.trace = trace
        self.error_action = error_action
        self.stepwise = stepwise
        self.model = None
        self.model_fit = None
        self.best_params = None
        
    def fit(self, data: Union[pd.Series, np.ndarray], **kwargs) -> 'AutoARIMA':
        """
        Fit the AutoARIMA model to the data
        
        Parameters:
        -----------
        data : pd.Series or np.ndarray
            Time series data to fit the model to
        **kwargs : dict
            Additional keyword arguments to pass to auto_arima
            
        Returns:
        --------
        self : AutoARIMA
            Fitted model instance
        """
        # Check stationarity if d is None
        if self.d is None and isinstance(data, pd.Series):
            is_stationary, _ = check_stationarity(data)
            if is_stationary:
                print("Data is stationary. Setting d=0.")
                self.d = 0
        
        self.model = auto_arima(
            data,
            start_p=self.start_p,
            max_p=self.max_p,
            start_q=self.start_q,
            max_q=self.max_q,
            d=self.d,
            start_P=self.start_P,
            max_P=self.max_P,
            start_Q=self.start_Q,
            max_Q=self.max_Q,
            max_D=self.max_D,
            seasonal=self.seasonal,
            m=self.m,
            information_criterion=self.information_criterion,
            trace=self.trace,
            error_action=self.error_action,
            stepwise=self.stepwise,
            **kwargs
        )
        
        # Store best parameters
        self.best_params = {
            'p': self.model.order[0],
            'd': self.model.order[1],
            'q': self.model.order[2]
        }
        
        if self.seasonal:
            self.best_params.update({
                'P': self.model.seasonal_order[0],
                'D': self.model.seasonal_order[1],
                'Q': self.model.seasonal_order[2],
                'm': self.model.seasonal_order[3]
            })
            
        return self
    
    def predict(self, n_periods: int = 1, return_conf_int: bool = False, alpha: float = 0.05) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate forecasts from the fitted model
        
        Parameters:
        -----------
        n_periods : int
            Number of periods to forecast
        return_conf_int : bool
            Whether to return confidence intervals
        alpha : float
            Significance level for confidence intervals
            
        Returns:
        --------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Forecast values and optionally confidence intervals
        """
        if self.model is None:
            raise ValueError("Model must be fit before forecasting.")
            
        if return_conf_int:
            return self.model.predict(n_periods=n_periods, return_conf_int=return_conf_int, alpha=alpha)
        else:
            return self.model.predict(n_periods=n_periods)
    
    def summary(self) -> None:
        """Print summary of the fitted model"""
        if self.model is None:
            raise ValueError("Model must be fit first.")
        
        print(self.model.summary())
        print(f"\nBest model parameters: {self.best_params}")
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot model diagnostics
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size for the diagnostics plots
        """
        if self.model is None:
            raise ValueError("Model must be fit first.")
        
        self.model.plot_diagnostics(figsize=figsize)
        plt.tight_layout()
        plt.show()


class ForecastARIMA:
    """
    Class for ARIMA model forecasting with specified parameters
    """
    
    def __init__(
        self, 
        p: int = 1, 
        d: int = 0, 
        q: int = 0,
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: Optional[str] = None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True
    ):
        """
        Initialize ForecastARIMA with specified parameters
        
        Parameters:
        -----------
        p : int
            AR order
        d : int
            Differencing order
        q : int
            MA order
        seasonal_order : Tuple[int, int, int, int]
            Seasonal components (P, D, Q, m)
        trend : Optional[str]
            Trend component ('n', 'c', 't', 'ct')
        enforce_stationarity : bool
            Whether to enforce stationarity for AR parameters
        enforce_invertibility : bool
            Whether to enforce invertibility for MA parameters
        """
        self.p = p
        self.d = d
        self.q = q
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model = None
        self.model_fit = None
        
    def fit(self, data: Union[pd.Series, np.ndarray], **kwargs) -> 'ForecastARIMA':
        """
        Fit the ARIMA model to the data
        
        Parameters:
        -----------
        data : pd.Series or np.ndarray
            Time series data to fit the model to
        **kwargs : dict
            Additional keyword arguments to pass to SARIMAX
            
        Returns:
        --------
        self : ForecastARIMA
            Fitted model instance
        """
        # Check if data is stationary if it's a pandas Series
        if isinstance(data, pd.Series) and self.d == 0:
            is_stationary, _ = check_stationarity(data)
            if not is_stationary:
                print("Warning: Data may not be stationary, but d=0 was specified.")
        
        # Check if seasonal components are non-zero
        has_seasonal = any(x > 0 for x in self.seasonal_order)
        
        # Use SARIMAX for both seasonal and non-seasonal models for consistency
        self.model = SARIMAX(
            data,
            order=(self.p, self.d, self.q),
            seasonal_order=self.seasonal_order if has_seasonal else None,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            **kwargs
        )
        
        self.model_fit = self.model.fit(disp=False)
        return self
    
    def predict(
        self, 
        n_periods: int = 1, 
        dynamic: bool = False, 
        return_conf_int: bool = False, 
        alpha: float = 0.05,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> Union[pd.Series, np.ndarray, Tuple[Union[pd.Series, np.ndarray], np.ndarray]]:
        """
        Generate predictions from the fitted model
        
        Parameters:
        -----------
        n_periods : int
            Number of periods to forecast (used if start/end not provided)
        dynamic : bool
            Whether to do dynamic forecasting
        return_conf_int : bool
            Whether to return confidence intervals
        alpha : float
            Significance level for confidence intervals
        start : Optional[int]
            Start index for prediction
        end : Optional[int]
            End index for prediction
            
        Returns:
        --------
        Union[pd.Series, np.ndarray, Tuple[Union[pd.Series, np.ndarray], np.ndarray]]
            Predicted values and optionally confidence intervals
        """
        if self.model_fit is None:
            raise ValueError("Model must be fit before forecasting.")
            
        # If start/end not provided, forecast n_periods ahead
        if start is None and end is None:
            start = self.model_fit.nobs
            end = start + n_periods - 1
            
        predictions = self.model_fit.get_prediction(start=start, end=end, dynamic=dynamic)
        
        if return_conf_int:
            pred_mean = predictions.predicted_mean
            conf_int = predictions.conf_int(alpha=alpha)
            return pred_mean, conf_int
        else:
            return predictions.predicted_mean
    
    def summary(self) -> None:
        """Print summary of the fitted model"""
        if self.model_fit is None:
            raise ValueError("Model must be fit first.")
        
        print(self.model_fit.summary())
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot model diagnostics
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size for the diagnostics plots
        """
        if self.model_fit is None:
            raise ValueError("Model must be fit first.")
        
        self.model_fit.plot_diagnostics(figsize=figsize)
        plt.tight_layout()
        plt.show()
    
    def evaluate(self, test_data: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate model performance against test data
        
        Parameters:
        -----------
        test_data : pd.Series or np.ndarray
            Actual values to compare predictions against
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        if self.model_fit is None:
            raise ValueError("Model must be fit first.")
        
        # Get predictions for the test period
        preds = self.predict(n_periods=len(test_data))
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(test_data, preds)),
            'mae': mean_absolute_error(test_data, preds),
            'mape': np.mean(np.abs((test_data - preds) / test_data)) * 100 if isinstance(test_data, np.ndarray) else 
                   np.mean(np.abs((test_data - preds) / test_data)) * 100
        }
        
        return metrics

if __name__ == "__main__":
    # Load your time series data
    data = load_time_series_parquet('path/to/your/data.parquet')

    # Split data into train and test
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Use AutoARIMA to automatically find the best parameters
    auto_model = AutoARIMA(seasonal=False)  # For monthly data with yearly seasonality
    auto_model.fit(train_data)
    auto_model.summary()

    # Or use ForecastARIMA with specific parameters
    # forecast_model = ForecastARIMA(p=2, d=1, q=2)
    # forecast_model.fit(train_data)

    # Generate forecasts
    predictions = auto_model.predict(n_periods=len(test_data))

    # Evaluate the model
    metrics = auto_model.evaluate(test_data)
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")

    # Plot diagnostics
    auto_model.plot_diagnostics()
=======

>>>>>>> main
