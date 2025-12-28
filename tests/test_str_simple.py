"""
Tests for simplified STR implementation.
"""

import pytest
import numpy as np
from strpy import (
    STR_decompose,
    AutoSTR_simple,
    moving_average_decompose,
    generate_synthetic_data
)
from strpy.simulations import rmse


def test_str_decompose_basic():
    """Test basic STR decomposition."""
    np.random.seed(42)

    # Generate simple data
    df = generate_synthetic_data(
        n=100,
        periods=(7,),
        gamma=0.1,
        random_seed=42
    )

    # Decompose
    result = STR_decompose(
        df['data'].values,
        seasonal_periods=[7],
        trend_lambda=100.0,
        seasonal_lambda=10.0
    )

    # Check outputs
    assert len(result.trend) == 100
    assert len(result.seasonal) == 1
    assert len(result.seasonal[0]) == 100
    assert len(result.remainder) == 100

    # Check reconstruction
    fitted = result.trend + result.seasonal[0]
    reconstruction_error = rmse(df['data'].values - fitted - result.remainder)
    assert reconstruction_error < 1e-10


def test_str_decompose_accuracy():
    """Test STR decomposition accuracy on known data."""
    np.random.seed(123)

    # Generate data with low noise
    df = generate_synthetic_data(
        n=200,
        periods=(7,),
        gamma=0.05,  # Low noise
        random_seed=123
    )

    # Decompose
    result = STR_decompose(
        df['data'].values,
        seasonal_periods=[7],
        trend_lambda=500.0,
        seasonal_lambda=50.0
    )

    # Should recover trend reasonably well
    trend_rmse = rmse(df['trend'].values - result.trend)
    assert trend_rmse < 2.0  # Reasonable threshold

    # Seasonal should be periodic
    seasonal = result.seasonal[0]
    # Check that seasonal pattern repeats
    diff_7 = np.abs(seasonal[7:] - seasonal[:-7])
    assert diff_7.max() < 0.5  # Should be fairly consistent


def test_str_multiple_seasonalities():
    """Test STR with multiple seasonal components."""
    np.random.seed(42)

    n = 365
    t = np.arange(n)

    # Create data with two seasonalities
    trend = 0.01 * t
    weekly = np.sin(2 * np.pi * t / 7)
    monthly = 0.5 * np.sin(2 * np.pi * t / 30)
    noise = 0.1 * np.random.randn(n)
    data = trend + weekly + monthly + noise

    # Decompose with both periods
    result = STR_decompose(
        data,
        seasonal_periods=[7, 30],
        trend_lambda=1000.0,
        seasonal_lambda=100.0
    )

    # Check we got two seasonal components
    assert len(result.seasonal) == 2

    # Check both are periodic
    for i, period in enumerate([7, 30]):
        seasonal = result.seasonal[i]
        # Seasonality should repeat
        if len(seasonal) > 2 * period:
            diff = np.abs(seasonal[period:2*period] - seasonal[:period])
            assert diff.max() < 1.0


def test_autostr_simple():
    """Test automatic STR parameter selection."""
    np.random.seed(42)

    df = generate_synthetic_data(
        n=100,
        periods=(7,),
        gamma=0.2,
        random_seed=42
    )

    # Run AutoSTR
    result = AutoSTR_simple(
        df['data'].values,
        seasonal_periods=[7],
        n_trials=5  # Few trials for speed
    )

    # Should find reasonable parameters
    assert result.params['trend_lambda'] > 0
    assert result.params['seasonal_lambda'] > 0

    # Should produce valid decomposition
    assert len(result.trend) == 100
    assert len(result.seasonal[0]) == 100

    # Remainder should be smaller than original data variance
    assert result.remainder.var() < df['data'].var()


def test_moving_average_decompose():
    """Test baseline moving average decomposition."""
    np.random.seed(42)

    # Generate data
    n = 100
    t = np.arange(n)
    trend = 0.01 * t
    seasonal = np.tile([1, 2, 3, 4, 5, 6, 7], n // 7 + 1)[:n]
    data = trend + seasonal + 0.1 * np.random.randn(n)

    # Decompose
    result = moving_average_decompose(data, period=7)

    # Check outputs
    assert 'trend' in result
    assert 'seasonal' in result
    assert 'remainder' in result

    assert len(result['trend']) == n
    assert len(result['seasonal']) == n
    assert len(result['remainder']) == n

    # Seasonal should be centered
    assert abs(result['seasonal'].mean()) < 0.1


def test_str_decompose_short_input():
    """Test STR with short input - should still work but may not be accurate."""
    # Very short series - will run but results may not be meaningful
    result = STR_decompose(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        seasonal_periods=[7]
    )

    # Should at least produce output
    assert result is not None
    assert len(result.trend) == 10


def test_str_result_attributes():
    """Test that STRResult has all expected attributes."""
    np.random.seed(42)

    df = generate_synthetic_data(n=50, periods=(7,), random_seed=42)

    result = STR_decompose(
        df['data'].values,
        seasonal_periods=[7]
    )

    # Check all attributes exist
    assert hasattr(result, 'data')
    assert hasattr(result, 'trend')
    assert hasattr(result, 'seasonal')
    assert hasattr(result, 'remainder')
    assert hasattr(result, 'fitted')
    assert hasattr(result, 'params')

    # Check fitted values
    assert result.fitted is not None
    assert len(result.fitted) == len(result.data)


def test_seasonal_centering():
    """Test that seasonal components are centered."""
    np.random.seed(42)

    df = generate_synthetic_data(n=140, periods=(7,), random_seed=42)

    result = STR_decompose(
        df['data'].values,
        seasonal_periods=[7]
    )

    # Seasonal component should have mean close to zero
    seasonal_mean = result.seasonal[0].mean()
    assert abs(seasonal_mean) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
