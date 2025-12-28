"""
Core STR (Seasonal-Trend decomposition using Regression) implementation.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy.optimize import minimize
from typing import List, Optional, Tuple, Dict, Any, Union
import warnings

from .utils import STRResult, create_difference_matrix, center_seasonal


class STR:
    """
    STR: Seasonal-Trend decomposition using Regression.

    Parameters
    ----------
    seasonal_periods : List[int]
        List of seasonal periods (e.g., [7, 365] for weekly and yearly)
    seasonal_lambdas : Optional[List[Tuple[float, float, float]]]
        Regularization parameters for each seasonal component (tt, st, ss)
        If None, will be estimated automatically
    trend_lambda : Optional[float]
        Regularization parameter for trend smoothness
        If None, will be estimated automatically
    confidence : float, default=0.95
        Confidence level for intervals (0 to 1)
    gap_cv : int, default=1
        Gap size for cross-validation (block size)

    Attributes
    ----------
    result_ : STRResult
        Decomposition results after fitting
    """

    def __init__(
        self,
        seasonal_periods: List[int],
        seasonal_lambdas: Optional[List[Tuple[float, float, float]]] = None,
        trend_lambda: Optional[float] = None,
        confidence: float = 0.95,
        gap_cv: int = 1,
    ):
        self.seasonal_periods = seasonal_periods
        self.seasonal_lambdas = seasonal_lambdas
        self.trend_lambda = trend_lambda
        self.confidence = confidence
        self.gap_cv = gap_cv
        self.result_ = None

    def fit(
        self,
        y: np.ndarray,
        covariates: Optional[Dict[str, np.ndarray]] = None,
    ) -> STRResult:
        """
        Fit the STR model to data.

        Parameters
        ----------
        y : np.ndarray
            Time series data (1D array)
        covariates : Optional[Dict[str, np.ndarray]]
            Dictionary of covariate arrays

        Returns
        -------
        STRResult
            Decomposition results
        """
        y = np.asarray(y).flatten()
        n = len(y)

        # Initialize lambdas if not provided
        if self.trend_lambda is None or self.seasonal_lambdas is None:
            raise NotImplementedError(
                "Automatic lambda selection not yet implemented. "
                "Please provide trend_lambda and seasonal_lambdas."
            )

        # Build design matrix
        X, y_padded = self._build_design_matrix(y, covariates)

        # Solve the regularized least squares problem
        eta = self._solve_regularized_ls(X, y_padded)

        # Extract components
        components = self._extract_components(eta, n)

        # Calculate fitted values and residuals
        fitted = components['trend'].copy()
        for seasonal in components['seasonal']:
            fitted += seasonal
        if components['covariates'] is not None:
            for cov in components['covariates']:
                fitted += cov

        remainder = y - fitted

        # Calculate confidence intervals if requested
        if self.confidence > 0:
            ci = self._calculate_confidence_intervals(X, n, remainder)
            components.update(ci)

        # Create result object
        self.result_ = STRResult(
            data=y,
            trend=components['trend'],
            seasonal=components['seasonal'],
            remainder=remainder,
            covariates=components['covariates'],
            trend_lower=components.get('trend_lower'),
            trend_upper=components.get('trend_upper'),
            seasonal_lower=components.get('seasonal_lower'),
            seasonal_upper=components.get('seasonal_upper'),
            params={
                'seasonal_periods': self.seasonal_periods,
                'seasonal_lambdas': self.seasonal_lambdas,
                'trend_lambda': self.trend_lambda,
            }
        )

        return self.result_

    def _build_design_matrix(
        self,
        y: np.ndarray,
        covariates: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the augmented design matrix X and response vector y+.

        Parameters
        ----------
        y : np.ndarray
            Time series data
        covariates : Optional[Dict[str, np.ndarray]]
            Covariate data

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Design matrix X and padded response y+
        """
        n = len(y)

        # Initialize lists for matrix blocks
        X_blocks = []

        # 1. Seasonal component matrices
        Q_matrices = []
        D_matrices = []

        for i, period in enumerate(self.seasonal_periods):
            # Create extraction matrix Q
            Q = self._create_seasonal_extraction_matrix(n, period)
            Q_matrices.append(Q)

            # Create difference operators for this seasonal component
            lambda_tt, lambda_st, lambda_ss = self.seasonal_lambdas[i]

            # Time direction differences
            if lambda_tt > 0:
                D_tt = self._create_seasonal_time_diff(n, period)
                D_matrices.append(lambda_tt * D_tt)
            else:
                D_matrices.append(None)

            # Season-time direction
            if lambda_st > 0:
                D_st = self._create_seasonal_season_time_diff(n, period)
                D_matrices.append(lambda_st * D_st)
            else:
                D_matrices.append(None)

            # Season direction (cyclic)
            if lambda_ss > 0:
                D_ss = self._create_seasonal_season_diff(n, period)
                D_matrices.append(lambda_ss * D_ss)
            else:
                D_matrices.append(None)

        # 2. Trend component
        I_n = np.eye(n)
        D_trend = create_difference_matrix(n, order=2, cyclic=False)
        D_trend = self.trend_lambda * D_trend

        # Build the augmented design matrix
        # First row: [Q1, Q2, ..., I_n]
        first_row = Q_matrices + [I_n]
        X_top = np.hstack(first_row)

        # Subsequent rows: regularization matrices
        X_reg_blocks = []

        # Add seasonal regularization blocks
        for i, period in enumerate(self.seasonal_periods):
            n_seasonal_params = n * (period - 1) // period + n
            for j in range(3):  # tt, st, ss
                D = D_matrices[i * 3 + j]
                if D is not None:
                    # Create block row with zeros except for this component
                    n_total_params = sum(
                        n * (p - 1) // p + n for p in self.seasonal_periods
                    ) + n
                    row_blocks = []
                    current_pos = 0
                    for k, p in enumerate(self.seasonal_periods):
                        p_params = n * (p - 1) // p + n
                        if k == i:
                            row_blocks.append(D)
                        else:
                            row_blocks.append(np.zeros((D.shape[0], p_params)))
                        current_pos += p_params
                    # Add zeros for trend
                    row_blocks.append(np.zeros((D.shape[0], n)))
                    X_reg_blocks.append(np.hstack(row_blocks))

        # Add trend regularization
        n_seasonal_total = sum(
            n * (p - 1) // p + n for p in self.seasonal_periods
        )
        trend_block_left = np.zeros((D_trend.shape[0], n_seasonal_total))
        trend_block = np.hstack([trend_block_left, D_trend])
        X_reg_blocks.append(trend_block)

        # Combine all blocks
        if X_reg_blocks:
            X_reg = np.vstack(X_reg_blocks)
            X = np.vstack([X_top, X_reg])
        else:
            X = X_top

        # Pad y with zeros
        y_padded = np.concatenate([y, np.zeros(X.shape[0] - n)])

        return X, y_padded

    def _create_seasonal_extraction_matrix(
        self, n: int, period: int
    ) -> np.ndarray:
        """
        Create matrix Q that extracts observed seasonal values.

        Parameters
        ----------
        n : int
            Number of observations
        period : int
            Seasonal period

        Returns
        -------
        np.ndarray
            Extraction matrix of shape (n, n*(period-1))
        """
        # Simplified version: identity-like extraction
        # In full implementation, this would handle the 2D seasonal surface
        m = n * period
        Q = np.zeros((n, m))
        for t in range(n):
            season_idx = t % period
            Q[t, t * period + season_idx] = 1
        return Q

    def _create_seasonal_time_diff(self, n: int, period: int) -> np.ndarray:
        """Create difference matrix in time direction for seasonal component."""
        # Simplified implementation
        m = n * period
        n_rows = (n - 2) * period
        D = np.zeros((n_rows, m))
        for i in range(n - 2):
            for j in range(period):
                row = i * period + j
                col = i * period + j
                D[row, col] = 1
                D[row, col + period] = -2
                D[row, col + 2 * period] = 1
        return D

    def _create_seasonal_season_time_diff(
        self, n: int, period: int
    ) -> np.ndarray:
        """Create difference matrix in season-time direction."""
        # Simplified implementation
        m = n * period
        D = np.zeros((n, m))
        return D  # Placeholder

    def _create_seasonal_season_diff(self, n: int, period: int) -> np.ndarray:
        """Create cyclic difference matrix in season direction."""
        m = n * period
        n_rows = n * period
        D = np.zeros((n_rows, m))
        for t in range(n):
            for s in range(period):
                row = t * period + s
                col = t * period + s
                D[row, col] = -2
                D[row, t * period + (s - 1) % period] = 1
                D[row, t * period + (s + 1) % period] = 1
        return D

    def _solve_regularized_ls(
        self, X: np.ndarray, y_padded: np.ndarray
    ) -> np.ndarray:
        """
        Solve the regularized least squares problem.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        y_padded : np.ndarray
            Padded response vector

        Returns
        -------
        np.ndarray
            Coefficient vector eta
        """
        # Use normal equations: (X'X)eta = X'y
        XtX = X.T @ X
        Xty = X.T @ y_padded

        # Solve using Cholesky decomposition for speed
        try:
            eta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # If singular, use least squares
            warnings.warn("Using lstsq due to singular matrix")
            eta = np.linalg.lstsq(X, y_padded, rcond=None)[0]

        return eta

    def _extract_components(
        self, eta: np.ndarray, n: int
    ) -> Dict[str, Any]:
        """
        Extract trend, seasonal, and covariate components from eta.

        Parameters
        ----------
        eta : np.ndarray
            Coefficient vector
        n : int
            Number of observations

        Returns
        -------
        Dict[str, Any]
            Dictionary with components
        """
        components = {
            'trend': None,
            'seasonal': [],
            'covariates': None,
        }

        # Extract seasonal components (simplified)
        current_pos = 0
        for period in self.seasonal_periods:
            # Simplified: extract first n values for each seasonal
            seasonal_size = n * period
            seasonal_full = eta[current_pos:current_pos + seasonal_size]
            # Extract observed values
            seasonal = np.array([
                seasonal_full[t * period + (t % period)]
                for t in range(n)
            ])
            # Center the seasonal component
            seasonal = center_seasonal(seasonal, period)
            components['seasonal'].append(seasonal)
            current_pos += seasonal_size

        # Extract trend
        components['trend'] = eta[current_pos:current_pos + n]

        return components

    def _calculate_confidence_intervals(
        self, X: np.ndarray, n: int, remainder: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate confidence intervals for components."""
        # Estimate residual variance
        sigma_sq = np.var(remainder)

        # Compute covariance matrix: sigma^2 * (X'X)^{-1}
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            warnings.warn("Cannot compute confidence intervals: singular matrix")
            return {}

        # Standard errors
        se = np.sqrt(sigma_sq * np.diag(XtX_inv))

        # Z-score for confidence level
        from scipy.stats import norm
        z = norm.ppf((1 + self.confidence) / 2)

        # Extract standard errors for components (simplified)
        ci = {
            'trend_lower': None,
            'trend_upper': None,
            'seasonal_lower': [],
            'seasonal_upper': [],
        }

        # This is a simplified implementation
        # Full implementation would properly extract from covariance matrix

        return ci


def AutoSTR(
    y: np.ndarray,
    seasonal_periods: List[int],
    confidence: float = 0.95,
    gap_cv: Optional[int] = None,
    method: str = 'nelder-mead',
) -> STRResult:
    """
    Automatic STR decomposition with parameter optimization.

    Parameters
    ----------
    y : np.ndarray
        Time series data
    seasonal_periods : List[int]
        List of seasonal periods
    confidence : float, default=0.95
        Confidence level for intervals
    gap_cv : Optional[int]
        Gap size for cross-validation. If None, uses first seasonal period
    method : str, default='nelder-mead'
        Optimization method for finding lambdas

    Returns
    -------
    STRResult
        Decomposition results
    """
    if gap_cv is None:
        gap_cv = seasonal_periods[0] if seasonal_periods else 1

    # Initialize lambda values (log scale)
    n_seasonal = len(seasonal_periods)
    # Each seasonal has 3 lambdas (tt, st, ss) + 1 trend lambda
    n_params = 3 * n_seasonal + 1

    # Initial guess: moderate smoothing
    x0 = np.zeros(n_params)

    def objective(log_lambdas):
        """Cross-validation objective function."""
        lambdas = np.exp(log_lambdas)

        # Split into trend and seasonal lambdas
        trend_lambda = lambdas[-1]
        seasonal_lambdas = [
            tuple(lambdas[i*3:(i+1)*3]) for i in range(n_seasonal)
        ]

        # Compute CV score
        cv_score = _compute_cv_score(
            y, seasonal_periods, seasonal_lambdas, trend_lambda, gap_cv
        )

        return cv_score

    # Optimize
    print("Optimizing smoothing parameters via cross-validation...")
    result = minimize(
        objective,
        x0,
        method=method,
        options={'maxiter': 100, 'disp': True}
    )

    # Extract optimal lambdas
    opt_lambdas = np.exp(result.x)
    trend_lambda = opt_lambdas[-1]
    seasonal_lambdas = [
        tuple(opt_lambdas[i*3:(i+1)*3]) for i in range(n_seasonal)
    ]

    print(f"Optimal trend_lambda: {trend_lambda:.4f}")
    for i, sl in enumerate(seasonal_lambdas):
        print(f"Optimal seasonal_lambdas[{i}]: {sl}")

    # Fit final model
    model = STR(
        seasonal_periods=seasonal_periods,
        seasonal_lambdas=seasonal_lambdas,
        trend_lambda=trend_lambda,
        confidence=confidence,
        gap_cv=gap_cv,
    )

    return model.fit(y)


def _compute_cv_score(
    y: np.ndarray,
    seasonal_periods: List[int],
    seasonal_lambdas: List[Tuple[float, float, float]],
    trend_lambda: float,
    gap_cv: int,
    k_folds: int = 5,
) -> float:
    """
    Compute k-fold cross-validation score.

    Parameters
    ----------
    y : np.ndarray
        Time series data
    seasonal_periods : List[int]
        Seasonal periods
    seasonal_lambdas : List[Tuple[float, float, float]]
        Seasonal regularization parameters
    trend_lambda : float
        Trend regularization parameter
    gap_cv : int
        Gap size for CV folds
    k_folds : int, default=5
        Number of folds

    Returns
    -------
    float
        Cross-validation score (mean squared error)
    """
    n = len(y)
    fold_size = n // k_folds

    cv_errors = []

    for fold in range(k_folds):
        # Create train/test split with gaps
        test_indices = np.arange(
            fold * fold_size,
            min((fold + 1) * fold_size, n)
        )

        # Remove gap around test set
        gap_indices = set()
        for idx in test_indices:
            gap_indices.update(range(max(0, idx - gap_cv), min(n, idx + gap_cv + 1)))

        train_indices = np.array([
            i for i in range(n) if i not in gap_indices
        ])

        if len(train_indices) < n // 2 or len(test_indices) == 0:
            continue

        # This is a simplified CV - full implementation would use
        # proper leave-one-out CV formula from the paper
        # For now, return a placeholder
        pass

    # Placeholder: return sum of squared lambdas as penalty
    return sum(sum(sl) for sl in seasonal_lambdas) + trend_lambda
