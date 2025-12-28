"""
Functions for generating synthetic time series data for testing STR decomposition.
"""

import numpy as np
from typing import Literal, Tuple, Optional, Dict
import pandas as pd


def generate_trend(
    n: int,
    trend_type: Literal["deterministic", "stochastic"] = "stochastic"
) -> np.ndarray:
    """
    Generate a trend component.

    Parameters
    ----------
    n : int
        Number of time points
    trend_type : {"deterministic", "stochastic"}
        Type of trend to generate

    Returns
    -------
    np.ndarray
        Trend component (standardized to mean 0, std 1)
    """
    if trend_type == "stochastic":
        # ARIMA(0,2,0): double integration of white noise
        trend = np.cumsum(np.cumsum(np.random.randn(2 * n)))[-n:]
    else:  # deterministic
        # Quadratic trend with random coefficients
        t = np.arange(n)
        n1 = np.random.randn()
        n2 = np.random.randn()
        trend = n1 * (t + n / 2 * (n2 - 1)) ** 2

    # Standardize
    trend = (trend - np.mean(trend)) / np.std(trend)
    return trend


def generate_seasonal(
    n: int,
    period: int,
    seasonal_type: Literal["deterministic", "stochastic"] = "stochastic",
    n_harmonics: int = 5,
) -> np.ndarray:
    """
    Generate a seasonal component.

    Parameters
    ----------
    n : int
        Number of time points
    period : int
        Seasonal period
    seasonal_type : {"deterministic", "stochastic"}
        Type of seasonality
    n_harmonics : int, default=5
        Number of Fourier harmonics (for deterministic type)

    Returns
    -------
    np.ndarray
        Seasonal component (standardized to mean 0, std 1)
    """
    if seasonal_type == "stochastic":
        # Generate smooth periodic pattern via double integration
        # Start with one period of random values
        base_season = np.random.randn(period)
        base_season = (base_season - np.mean(base_season)) / np.std(base_season)

        # Replicate to cover full length
        full_season = np.tile(base_season, n // period + 1)[:n]

        # Integrate twice to smooth
        smooth = np.cumsum(np.cumsum(full_season))
        smooth = (smooth - np.mean(smooth)) / np.std(smooth)

        seasonal = smooth
    else:  # deterministic
        # Fourier series with random coefficients
        t = np.arange(n)
        seasonal = np.zeros(n)

        for k in range(1, n_harmonics + 1):
            a_k = np.random.randn()
            b_k = np.random.randn()
            seasonal += a_k * np.sin(2 * np.pi * k * t / period)
            seasonal += b_k * np.cos(2 * np.pi * k * t / period)

    # Standardize
    seasonal = (seasonal - np.mean(seasonal)) / np.std(seasonal)
    return seasonal


def generate_synthetic_data(
    n: int = 1096,
    periods: Tuple[int, ...] = (7, 365),
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.25,
    data_type: Literal["deterministic", "stochastic"] = "stochastic",
    n_harmonics: int = 5,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic time series with trend, multiple seasonal components, and noise.

    The model is: y_t = T_t + alpha * S^(1)_t + beta * S^(2)_t + gamma * R_t

    Parameters
    ----------
    n : int, default=1096
        Number of time points (default is 3 years of daily data)
    periods : Tuple[int, ...], default=(7, 365)
        Seasonal periods (e.g., weekly=7, yearly=365)
    alpha : float, default=1.0
        Weight for first seasonal component
    beta : float, default=1.0
        Weight for second seasonal component
    gamma : float, default=0.25
        Weight for noise component
    data_type : {"deterministic", "stochastic"}, default="stochastic"
        Type of data generation process
    n_harmonics : int, default=5
        Number of Fourier harmonics for deterministic seasonality
    random_seed : Optional[int]
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: trend, seasonal_1, seasonal_2, ..., remainder, data
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate trend
    trend = generate_trend(n, data_type)

    # Generate seasonal components
    seasonals = []
    weights = [alpha, beta] if len(periods) == 2 else [alpha] * len(periods)

    for i, period in enumerate(periods):
        seasonal = generate_seasonal(n, period, data_type, n_harmonics)
        seasonals.append(weights[i] * seasonal)

    # Generate noise
    remainder = gamma * np.random.randn(n)

    # Combine components
    data = trend + sum(seasonals) + remainder

    # Create DataFrame
    df = pd.DataFrame({
        'trend': trend,
        'remainder': remainder,
        'data': data,
    })

    # Add seasonal components
    for i, seasonal in enumerate(seasonals):
        df[f'seasonal_{i+1}'] = seasonal

    return df


def compute_decomposition_errors(
    true_components: pd.DataFrame,
    estimated_components: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compute errors between true and estimated components.

    Parameters
    ----------
    true_components : pd.DataFrame
        DataFrame with true components (from generate_synthetic_data)
    estimated_components : Dict[str, np.ndarray]
        Dictionary with estimated components

    Returns
    -------
    pd.DataFrame
        DataFrame with errors for each component
    """
    errors = {}

    # Trend error
    if 'trend' in estimated_components:
        errors['trend'] = true_components['trend'].values - estimated_components['trend']

    # Seasonal errors
    for i in range(10):  # Check up to 10 seasonal components
        key = f'seasonal_{i+1}'
        if key in true_components.columns:
            if 'seasonal' in estimated_components and len(estimated_components['seasonal']) > i:
                errors[key] = (
                    true_components[key].values - estimated_components['seasonal'][i]
                )

    # Remainder error
    if 'remainder' in estimated_components:
        errors['remainder'] = (
            true_components['remainder'].values - estimated_components['remainder']
        )

    return pd.DataFrame(errors)


def rmse(errors: np.ndarray) -> float:
    """
    Compute root mean squared error.

    Parameters
    ----------
    errors : np.ndarray
        Array of errors

    Returns
    -------
    float
        RMSE value
    """
    return np.sqrt(np.mean(errors ** 2))
