"""
Simplified STR (Seasonal-Trend decomposition using Regression) implementation.

This is a working implementation with simplified assumptions:
- Uses standard seasonal dummy variables approach
- Regularization via ridge regression
- Suitable for basic seasonal decomposition
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Optional, Tuple
import warnings

from .utils import STRResult, create_difference_matrix, center_seasonal


def STR_decompose(
    y: np.ndarray,
    seasonal_periods: List[int],
    trend_lambda: float = 1000.0,
    seasonal_lambda: float = 100.0,
    robust: bool = False,
) -> STRResult:
    """
    Simplified STR decomposition.

    Parameters
    ----------
    y : np.ndarray
        Time series data (1D array)
    seasonal_periods : List[int]
        List of seasonal periods (e.g., [7, 365])
    trend_lambda : float, default=1000.0
        Smoothing parameter for trend
    seasonal_lambda : float, default=100.0
        Smoothing parameter for seasonal components
    robust : bool, default=False
        Use robust regression (not yet implemented)

    Returns
    -------
    STRResult
        Decomposition results
    """
    y = np.asarray(y).flatten()
    n = len(y)

    # Build design matrix
    X_components = []

    # 1. Trend: Identity matrix (will be smoothed via regularization)
    X_trend = np.eye(n)
    X_components.append(X_trend)

    # 2. Seasonal components: Fourier basis
    seasonal_matrices = []
    for period in seasonal_periods:
        X_seasonal = _create_seasonal_matrix(n, period)
        X_components.append(X_seasonal)
        seasonal_matrices.append(X_seasonal)

    # Combine all components
    X = np.hstack(X_components)

    # Build regularization matrix
    D_components = []

    # Trend regularization (second differences)
    D_trend = create_difference_matrix(n, order=2, cyclic=False)
    D_trend_padded = np.zeros((D_trend.shape[0], X.shape[1]))
    D_trend_padded[:, :n] = D_trend
    D_components.append(trend_lambda * D_trend_padded)

    # Seasonal regularization (second differences on the seasonal pattern itself)
    offset = n
    for i, period in enumerate(seasonal_periods):
        n_seasonal_cols = seasonal_matrices[i].shape[1]

        # Create difference operator for seasonal smoothness
        # Apply difference operator to the PRODUCT: D @ (X_seasonal @ beta_seasonal)
        # This is equivalent to: (D @ X_seasonal) @ beta_seasonal
        X_seas = seasonal_matrices[i]
        D_seasonal_pattern = create_difference_matrix(n, order=2, cyclic=True)
        D_seasonal_transformed = D_seasonal_pattern @ X_seas

        # Pad to full width
        D_seasonal_padded = np.zeros((D_seasonal_transformed.shape[0], X.shape[1]))
        D_seasonal_padded[:, offset:offset + n_seasonal_cols] = D_seasonal_transformed
        D_components.append(seasonal_lambda * D_seasonal_padded)

        offset += n_seasonal_cols

    # Stack regularization matrices
    D = np.vstack(D_components)

    # Augmented system: [X; D] @ beta = [y; 0]
    X_aug = np.vstack([X, D])
    y_aug = np.concatenate([y, np.zeros(D.shape[0])])

    # Solve via normal equations
    try:
        beta = np.linalg.lstsq(X_aug, y_aug, rcond=None)[0]
    except np.linalg.LinAlgError:
        warnings.warn("Using ridge regression due to singular matrix")
        # Fall back to explicit ridge regression
        XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
        DtD = D.T @ D
        beta = np.linalg.solve(XtX + DtD, X.T @ y)

    # Extract components
    trend = beta[:n]

    seasonals = []
    offset = n
    for i, period in enumerate(seasonal_periods):
        n_cols = seasonal_matrices[i].shape[1]
        seasonal_beta = beta[offset:offset + n_cols]
        seasonal = seasonal_matrices[i] @ seasonal_beta
        # Center the seasonal component
        seasonal = center_seasonal(seasonal, period)
        seasonals.append(seasonal)
        offset += n_cols

    # Calculate remainder
    fitted = trend + sum(seasonals)
    remainder = y - fitted

    # Create result
    result = STRResult(
        data=y,
        trend=trend,
        seasonal=seasonals,
        remainder=remainder,
        params={
            'seasonal_periods': seasonal_periods,
            'trend_lambda': trend_lambda,
            'seasonal_lambda': seasonal_lambda,
        }
    )

    return result


def _create_seasonal_matrix(n: int, period: int, n_harmonics: Optional[int] = None) -> np.ndarray:
    """
    Create Fourier seasonal design matrix.

    Parameters
    ----------
    n : int
        Number of observations
    period : int
        Seasonal period
    n_harmonics : Optional[int]
        Number of Fourier harmonics to use
        If None, uses min(10, period // 2)

    Returns
    -------
    np.ndarray
        Design matrix of shape (n, 2*n_harmonics)
    """
    if n_harmonics is None:
        n_harmonics = min(10, period // 2)

    t = np.arange(n)
    X_seasonal = []

    for k in range(1, n_harmonics + 1):
        # Sin and cos components
        X_seasonal.append(np.sin(2 * np.pi * k * t / period))
        X_seasonal.append(np.cos(2 * np.pi * k * t / period))

    return np.column_stack(X_seasonal)


def AutoSTR_simple(
    y: np.ndarray,
    seasonal_periods: List[int],
    n_trials: int = 10,
) -> STRResult:
    """
    Automatic STR with simple parameter search.

    Parameters
    ----------
    y : np.ndarray
        Time series data
    seasonal_periods : List[int]
        Seasonal periods
    n_trials : int, default=10
        Number of random parameter combinations to try

    Returns
    -------
    STRResult
        Best decomposition based on AIC-like criterion
    """
    y = np.asarray(y).flatten()
    n = len(y)

    best_score = np.inf
    best_result = None

    print(f"Searching for optimal parameters ({n_trials} trials)...")

    # Try different lambda combinations
    for trial in range(n_trials):
        # Random search in log space
        trend_lambda = 10 ** np.random.uniform(1, 4)
        seasonal_lambda = 10 ** np.random.uniform(0, 3)

        try:
            result = STR_decompose(
                y,
                seasonal_periods,
                trend_lambda=trend_lambda,
                seasonal_lambda=seasonal_lambda
            )

            # Scoring: prefer models with low remainder variance
            # but penalize extreme smoothing
            rss = np.sum(result.remainder ** 2)

            # Check if seasonal component has meaningful variance
            seasonal_var = result.seasonal[0].var()
            if seasonal_var < 0.01:  # Seasonal is too flat
                score = np.inf  # Reject this solution
            else:
                # Use BIC-like criterion: log-likelihood + complexity penalty
                log_likelihood = -n/2 * np.log(2*np.pi*rss/n) - n/2
                # Penalty based on smoothness (higher lambda = simpler model)
                complexity = -np.log(trend_lambda + 1) - np.log(seasonal_lambda + 1)
                score = -log_likelihood + complexity

            if score < best_score:
                best_score = score
                best_result = result
                print(f"  Trial {trial+1}: trend_λ={trend_lambda:.1f}, "
                      f"seasonal_λ={seasonal_lambda:.1f}, score={score:.2f} ✓")
        except Exception as e:
            if trial < 3:  # Only print first few errors
                print(f"  Trial {trial+1}: Failed ({str(e)[:50]})")
            continue

    if best_result is None:
        # Fall back to default parameters
        print("  Using default parameters")
        best_result = STR_decompose(y, seasonal_periods)

    print(f"\nOptimal parameters found:")
    print(f"  trend_lambda: {best_result.params['trend_lambda']:.1f}")
    print(f"  seasonal_lambda: {best_result.params['seasonal_lambda']:.1f}")

    return best_result


def moving_average_decompose(
    y: np.ndarray,
    period: int
) -> dict:
    """
    Simple moving average decomposition (baseline method).

    Parameters
    ----------
    y : np.ndarray
        Time series data
    period : int
        Seasonal period

    Returns
    -------
    dict
        Dictionary with 'trend', 'seasonal', 'remainder' components
    """
    import pandas as pd

    y = np.asarray(y).flatten()
    n = len(y)

    # Centered moving average for trend
    if period % 2 == 0:
        # Even period: use 2x(period) MA
        ma = pd.Series(y).rolling(window=period, center=True).mean()
        trend = ma.rolling(window=2, center=True).mean().values
    else:
        # Odd period: simple centered MA
        trend = pd.Series(y).rolling(window=period, center=True).mean().values

    # Detrend
    detrended = y - np.nan_to_num(trend, nan=0)

    # Seasonal: average by position in cycle
    seasonal = np.zeros(n)
    for i in range(period):
        indices = np.arange(i, n, period)
        # Use median for robustness
        seasonal[indices] = np.nanmedian(detrended[indices])

    # Center seasonal
    seasonal = seasonal - np.mean(seasonal)

    # Remainder
    remainder = y - np.nan_to_num(trend, nan=0) - seasonal

    return {
        'trend': np.nan_to_num(trend, nan=0),
        'seasonal': seasonal,
        'remainder': remainder
    }
