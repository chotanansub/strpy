"""
Utility functions and data structures for STR decomposition.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class STRResult:
    """
    Container for STR decomposition results.

    Attributes
    ----------
    data : np.ndarray
        Original time series data
    trend : np.ndarray
        Trend component
    seasonal : List[np.ndarray]
        List of seasonal components
    remainder : np.ndarray
        Remainder component (residuals)
    covariates : Optional[List[np.ndarray]]
        Covariate effects if covariates were included
    trend_lower : Optional[np.ndarray]
        Lower confidence bound for trend
    trend_upper : Optional[np.ndarray]
        Upper confidence bound for trend
    seasonal_lower : Optional[List[np.ndarray]]
        Lower confidence bounds for seasonal components
    seasonal_upper : Optional[List[np.ndarray]]
        Upper confidence bounds for seasonal components
    fitted : np.ndarray
        Fitted values (data - remainder)
    params : Dict[str, Any]
        Parameters used for decomposition
    """
    data: np.ndarray
    trend: np.ndarray
    seasonal: List[np.ndarray]
    remainder: np.ndarray
    covariates: Optional[List[np.ndarray]] = None
    trend_lower: Optional[np.ndarray] = None
    trend_upper: Optional[np.ndarray] = None
    seasonal_lower: Optional[List[np.ndarray]] = None
    seasonal_upper: Optional[List[np.ndarray]] = None
    fitted: Optional[np.ndarray] = None
    params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Calculate fitted values if not provided."""
        if self.fitted is None:
            self.fitted = self.trend + sum(self.seasonal)
            if self.covariates is not None:
                self.fitted += sum(self.covariates)

    def plot(self, figsize=(12, 10)):
        """
        Plot the decomposition components.

        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        """
        import matplotlib.pyplot as plt

        n_components = 2 + len(self.seasonal)
        if self.covariates:
            n_components += len(self.covariates)
        n_components += 1  # remainder

        fig, axes = plt.subplots(n_components, 1, figsize=figsize, sharex=True)

        # Original data
        axes[0].plot(self.data, 'k-', linewidth=0.8)
        axes[0].set_ylabel('Data')
        axes[0].set_title('STR Decomposition')
        axes[0].grid(True, alpha=0.3)

        # Trend with confidence intervals
        axes[1].plot(self.trend, 'r-', linewidth=1.2)
        if self.trend_lower is not None and self.trend_upper is not None:
            axes[1].fill_between(
                range(len(self.trend)),
                self.trend_lower,
                self.trend_upper,
                alpha=0.3,
                color='blue'
            )
        axes[1].set_ylabel('Trend')
        axes[1].grid(True, alpha=0.3)

        # Seasonal components
        for i, seasonal in enumerate(self.seasonal):
            axes[2 + i].plot(seasonal, 'g-', linewidth=0.8)
            if (self.seasonal_lower is not None and
                self.seasonal_upper is not None):
                axes[2 + i].fill_between(
                    range(len(seasonal)),
                    self.seasonal_lower[i],
                    self.seasonal_upper[i],
                    alpha=0.3,
                    color='blue'
                )
            axes[2 + i].set_ylabel(f'Seasonal {i+1}')
            axes[2 + i].grid(True, alpha=0.3)

        # Covariates if present
        offset = 2 + len(self.seasonal)
        if self.covariates:
            for i, cov in enumerate(self.covariates):
                axes[offset + i].plot(cov, 'b-', linewidth=0.8)
                axes[offset + i].set_ylabel(f'Covariate {i+1}')
                axes[offset + i].grid(True, alpha=0.3)
            offset += len(self.covariates)

        # Remainder
        axes[offset].plot(self.remainder, 'gray', linewidth=0.5, alpha=0.7)
        axes[offset].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        axes[offset].set_ylabel('Remainder')
        axes[offset].set_xlabel('Time')
        axes[offset].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes


def create_difference_matrix(n: int, order: int = 2, cyclic: bool = False) -> np.ndarray:
    """
    Create a difference operator matrix.

    Parameters
    ----------
    n : int
        Size of the vector to be differenced
    order : int, default=2
        Order of differencing (1 or 2)
    cyclic : bool, default=False
        Whether to use cyclic differences (for seasonal patterns)

    Returns
    -------
    np.ndarray
        Difference matrix of shape (n-order, n) or (n, n) if cyclic
    """
    if order == 1:
        if cyclic:
            D = np.zeros((n, n))
            for i in range(n):
                D[i, i] = 1
                D[i, (i + 1) % n] = -1
        else:
            D = np.zeros((n - 1, n))
            for i in range(n - 1):
                D[i, i] = 1
                D[i, i + 1] = -1
    elif order == 2:
        if cyclic:
            D = np.zeros((n, n))
            for i in range(n):
                D[i, (i - 1) % n] = 1
                D[i, i] = -2
                D[i, (i + 1) % n] = 1
        else:
            D = np.zeros((n - 2, n))
            for i in range(n - 2):
                D[i, i] = 1
                D[i, i + 1] = -2
                D[i, i + 2] = 1
    else:
        raise ValueError("Only order 1 and 2 differences are supported")

    return D


def center_seasonal(seasonal: np.ndarray, period: int) -> np.ndarray:
    """
    Center seasonal component to have zero mean.

    Parameters
    ----------
    seasonal : np.ndarray
        Seasonal component
    period : int
        Seasonal period (not used in simple centering, kept for compatibility)

    Returns
    -------
    np.ndarray
        Centered seasonal component
    """
    # Simply subtract the overall mean to ensure seasonal has mean zero
    # This prevents confounding with the trend
    centered = seasonal.astype(float).copy()
    centered -= centered.mean()

    return centered
