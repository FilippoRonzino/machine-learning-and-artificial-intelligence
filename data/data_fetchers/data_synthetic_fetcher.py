import numpy as np
import os
import pandas as pd

def generate_harmonic_timeseries(
    T: int = 100, dt: float = 1,
    A1: float = 1.0, B1: float = 0.0, T1: float = 20.0, phi1: float = 0.0,
    A2: float = 0.5, B2: float = 0.0, T2: float = 5.0, phi2: float = 0.0
) -> np.ndarray:
    """
    Generate a harmonic time series.

    :param T: Total number of time steps.
    :param dt: Time step size.
    :param A1: Amplitude of the first harmonic component.
    :param B1: Linear trend of the first harmonic component.
    :param T1: Period of the first harmonic component.
    :param phi1: Phase shift of the first harmonic component.
    :param A2: Amplitude of the second harmonic component.
    :param B2: Linear trend of the second harmonic component.
    :param T2: Period of the second harmonic component.
    :param phi2: Phase shift of the second harmonic component.
    :return: s_t: The generated harmonic time series as a numpy array.
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if T1 <= 0:
        raise ValueError("T1 (period of first harmonic) must be positive.")
    if T2 <= 0:
        raise ValueError("T2 (period of second harmonic) must be positive.")

    t_values = np.arange(1, T + 1) * dt

    s1 = (A1 + B1 * t_values) * np.sin(2 * np.pi * t_values / T1 + phi1)
    s2 = (A2 + B2 * t_values) * np.sin(2 * np.pi * t_values / T2 + phi2)

    s_t = s1 + s2
   
    return s_t

def generate_ou_timeseries(
    T: int = 100, dt: float = 1.0,
    mu: float = 0.0, gamma: float = 0.1, sigma: float = 1.0,
    s0: float = 0.0
) -> np.ndarray:
    """
    Generate a time series based on the Ornstein-Uhlenbeck (OU) process.

    :param T: Total number of time steps.
    :param dt: Time step size.
    :param mu: Long-term mean of the process.
    :param gamma: Mean reversion rate (how fast the process reverts to the mean).
    :param sigma: Volatility parameter (standard deviation of the process noise).
    :param s0: Initial value of the time series at time t = 0.
    :return: s: The generated Ornstein-Uhlenbeck time series as a numpy array of length T.
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if gamma <= 0:
        raise ValueError("gamma must be positive to avoid division by zero or negative reversion rates.")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative.")
    
    s = np.zeros(T)
    s[0] = s0
    
    for i in range(1, T):
        t = i * dt
        exp_term = np.exp(-gamma * t)
        mean = mu + (s[i-1] - mu) * exp_term
        variance = (sigma ** 2) / (2 * gamma) * (1 - np.exp(-2 * gamma * t))
        s[i] = np.random.normal(loc=mean, scale=np.sqrt(variance))

    return s


def generate_harmonic_dataset(
    n_series: int = 60000,
    T: int = 100,
    dt: float = 1.0
) -> np.ndarray:
    """
    Generate a dataset of harmonic time series using randomized parameters.

    :param n_series: Number of time series to generate.
    :param T: Number of time steps in each time series.
    :param dt: Time step size.
    :return: dataset: Array of shape (n_series, T) containing the generated time series.
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if n_series < 0:
        raise ValueError("n_series cannot be negative.")

    dataset = np.zeros((n_series, T))

    for i in range(n_series):
        A1 = np.random.normal(loc=1.0, scale=0.5)
        A2 = np.random.normal(loc=1.0, scale=0.5)

        B1 = np.random.uniform(low=-1/T, high=1/T)
        B2 = np.random.uniform(low=-1/T, high=1/T)

        T1 = np.random.normal(loc=T/5, scale=T/10)  
        T2 = np.random.normal(loc=T, scale=T/2)     

        phi1 = np.random.uniform(low=0, high=2 * np.pi)
        phi2 = np.random.uniform(low=0, high=2 * np.pi)

        s_t = generate_harmonic_timeseries(
            T=T, dt=dt,
            A1=A1, B1=B1, T1=T1, phi1=phi1,
            A2=A2, B2=B2, T2=T2, phi2=phi2
        )

        dataset[i] = s_t

    return dataset

def generate_ou_dataset(
        n_series: int = 60000,
        T: int = 100,
        dt:float = 1,
) -> np.ndarray:
    """
    Generate a dataset of Ornstein-Uhlenbeck time series using randomized parameters.

    :param n_series: Number of time series to generate.
    :param T: Number of time steps in each time series.
    :param dt: Time step size.
    :return: dataset: Array of shape (n_series, T) containing the generated time series.
    :raises ValueError: If T, dt, or n_series are not positive.
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if n_series < 0:
        raise ValueError("n_series cannot be negative.")

    dataset = np.zeros((n_series, T))

    for i in range(n_series):
        mu = np.random.normal(loc=0.0, scale=1.0)
        gamma = np.random.normal(loc=8e-8, scale=4e-8)  # Mean reversion rate
        sigma = np.random.normal(loc=1e-2, scale=5e-3)  # Volatility
        s0 = np.random.normal(loc=0.0, scale=1.0)       # Initial value

        s_t = generate_ou_timeseries(T=T, dt=dt, mu=mu, gamma=gamma, sigma=sigma, s0=s0)
        dataset[i] = s_t

    return dataset


if __name__ == "__main__":
    os.makedirs("data/data_storage/harmonic_ou_parquets", exist_ok=True)
    
    n_series = 60000  # Number of time series
    T = 80  # Number of time steps
    dt = 1.0  # Time step size
    
    print("Generating harmonic dataset...")
    harmonic_dataset = generate_harmonic_dataset(n_series=n_series, T=T, dt=dt)
    
    print("Generating Ornstein-Uhlenbeck dataset...")
    ou_dataset = generate_ou_dataset(n_series=n_series, T=T, dt=dt)
    
    print("Saving datasets to Parquet files...")
    
    harmonic_df = pd.DataFrame(harmonic_dataset)
    harmonic_df.to_parquet("data/data_storage/harmonic_ou_parquets/harmonic_dataset.parquet", index=False)
    
    ou_df = pd.DataFrame(ou_dataset)
    ou_df.to_parquet("data/data_storage/harmonic_ou_parquets/ou_dataset.parquet", index=False)
    
    print("Datasets successfully saved to 'data/data_storage/harmonic_ou_parquets' directory.")