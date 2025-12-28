"""
Tests for utility functions.
"""

import pytest
import numpy as np
from strpy.utils import create_difference_matrix, center_seasonal


def test_create_difference_matrix_order1():
    """Test first-order difference matrix."""
    n = 5
    D = create_difference_matrix(n, order=1, cyclic=False)

    # Check shape
    assert D.shape == (n - 1, n)

    # Check values
    expected = np.array([
        [1, -1, 0, 0, 0],
        [0, 1, -1, 0, 0],
        [0, 0, 1, -1, 0],
        [0, 0, 0, 1, -1],
    ])
    np.testing.assert_array_equal(D, expected)


def test_create_difference_matrix_order2():
    """Test second-order difference matrix."""
    n = 5
    D = create_difference_matrix(n, order=2, cyclic=False)

    # Check shape
    assert D.shape == (n - 2, n)

    # Check values
    expected = np.array([
        [1, -2, 1, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 1, -2, 1],
    ])
    np.testing.assert_array_equal(D, expected)


def test_create_difference_matrix_cyclic():
    """Test cyclic difference matrix."""
    n = 4
    D = create_difference_matrix(n, order=2, cyclic=True)

    # Check shape (cyclic has same dimension)
    assert D.shape == (n, n)

    # Check that it wraps around
    # D @ x should give second differences with wrap-around
    x = np.array([1, 2, 3, 4])
    result = D @ x
    # Each element should be: x[i-1] - 2*x[i] + x[i+1] (with wrap)
    expected = np.array([
        4 - 2*1 + 2,  # x[3] - 2*x[0] + x[1]
        1 - 2*2 + 3,  # x[0] - 2*x[1] + x[2]
        2 - 2*3 + 4,  # x[1] - 2*x[2] + x[3]
        3 - 2*4 + 1,  # x[2] - 2*x[3] + x[0]
    ])
    np.testing.assert_array_equal(result, expected)


def test_center_seasonal():
    """Test seasonal centering."""
    # Create a seasonal pattern
    period = 7
    seasonal = np.array([1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7])

    centered = center_seasonal(seasonal, period)

    # Check that mean within each season position is zero
    for i in range(period):
        indices = np.arange(i, len(centered), period)
        season_mean = centered[indices].mean()
        assert abs(season_mean) < 1e-10


def test_center_seasonal_preserves_pattern():
    """Test that centering preserves the pattern shape."""
    period = 4
    # Create pattern with known structure
    base = np.array([10, 20, 30, 40])
    seasonal = np.tile(base, 3)

    centered = center_seasonal(seasonal, period)

    # Centered values at same season position should be equal
    for i in range(period):
        indices = np.arange(i, len(centered), period)
        values = centered[indices]
        # All values at same position should be equal (after centering)
        assert np.allclose(values, values[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
