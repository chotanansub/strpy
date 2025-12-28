"""
Tests for simulation functions.
"""

import pytest
import numpy as np
import pandas as pd
from strpy.simulations import (
    generate_trend,
    generate_seasonal,
    generate_synthetic_data,
    rmse,
)


def test_generate_trend_stochastic():
    """Test stochastic trend generation."""
    np.random.seed(42)
    n = 100
    trend = generate_trend(n, trend_type="stochastic")

    assert len(trend) == n
    assert isinstance(trend, np.ndarray)
    # Check standardization
    assert abs(trend.mean()) < 0.1
    assert abs(trend.std() - 1.0) < 0.1


def test_generate_trend_deterministic():
    """Test deterministic trend generation."""
    np.random.seed(42)
    n = 100
    trend = generate_trend(n, trend_type="deterministic")

    assert len(trend) == n
    assert isinstance(trend, np.ndarray)
    # Check standardization
    assert abs(trend.mean()) < 0.1
    assert abs(trend.std() - 1.0) < 0.1


def test_generate_seasonal_stochastic():
    """Test stochastic seasonal generation."""
    np.random.seed(42)
    n = 365
    period = 7
    seasonal = generate_seasonal(n, period, seasonal_type="stochastic")

    assert len(seasonal) == n
    assert isinstance(seasonal, np.ndarray)
    # Check standardization
    assert abs(seasonal.mean()) < 0.1
    assert abs(seasonal.std() - 1.0) < 0.1


def test_generate_seasonal_deterministic():
    """Test deterministic seasonal generation."""
    np.random.seed(42)
    n = 365
    period = 7
    seasonal = generate_seasonal(
        n, period, seasonal_type="deterministic", n_harmonics=3
    )

    assert len(seasonal) == n
    assert isinstance(seasonal, np.ndarray)
    # Check standardization
    assert abs(seasonal.mean()) < 0.1
    assert abs(seasonal.std() - 1.0) < 0.1


def test_generate_synthetic_data():
    """Test synthetic data generation."""
    np.random.seed(42)

    df = generate_synthetic_data(
        n=100,
        periods=(7, 28),
        alpha=1.0,
        beta=0.5,
        gamma=0.2,
        data_type="stochastic",
        random_seed=42
    )

    # Check output type and shape
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100

    # Check required columns
    required_cols = ['trend', 'seasonal_1', 'seasonal_2', 'remainder', 'data']
    for col in required_cols:
        assert col in df.columns

    # Check data is combination of components
    reconstructed = (
        df['trend'] + df['seasonal_1'] + df['seasonal_2'] + df['remainder']
    )
    np.testing.assert_array_almost_equal(df['data'].values, reconstructed.values)


def test_generate_synthetic_data_reproducibility():
    """Test that random seed produces reproducible results."""
    df1 = generate_synthetic_data(n=50, random_seed=123)
    df2 = generate_synthetic_data(n=50, random_seed=123)

    np.testing.assert_array_equal(df1['data'].values, df2['data'].values)
    np.testing.assert_array_equal(df1['trend'].values, df2['trend'].values)


def test_rmse():
    """Test RMSE calculation."""
    errors = np.array([1, -1, 2, -2])
    result = rmse(errors)
    expected = np.sqrt(2.5)
    assert abs(result - expected) < 1e-10


def test_rmse_zero():
    """Test RMSE with zero errors."""
    errors = np.zeros(10)
    result = rmse(errors)
    assert result == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
