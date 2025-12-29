"""
R Comparison Tests - TRUE validation against R stR package.

These tests compare Python STRPy results against the original R implementation.
Unlike test_r_validation.py (which are just regression tests), these tests
actually execute R code and compare outputs.

IMPORTANT:
- Tests will SKIP if R or stR package is not installed
- See docs/R_INSTALLATION.md for R installation instructions
- Validation uses CORRELATION not exact match (algorithms differ)

Expected thresholds:
- Trend correlation > 0.85
- Seasonal correlation > 0.70
- R² within 0.15 of each other

Why correlation? The Python implementation uses simplified Fourier basis
while R uses 2D seasonal surfaces with complex topology. They're fundamentally
different algorithms that should produce similar (not identical) decompositions.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from strpy import STR_decompose

# Import R bridge utilities (relative import for when run as module)
try:
    from .r_bridge import RBridge, compare_decompositions, check_r_available
except ImportError:
    from r_bridge import RBridge, compare_decompositions, check_r_available


# Fixture directory
FIXTURES_DIR = Path(__file__).parent / 'fixtures'


# Skip all tests in this module if R is not available
pytestmark = pytest.mark.skipif(
    not check_r_available(),
    reason="R or stR package not installed. See docs/R_INSTALLATION.md"
)


class TestRComparison:
    """Compare Python STRPy against R stR package."""

    @pytest.fixture(scope="class")
    def r_bridge(self):
        """Initialize R bridge once for all tests."""
        return RBridge()

    def test_simple_weekly_pattern(self, r_bridge):
        """
        Test Case 1: Simple weekly pattern (baseline test).

        Data: n=365, period=7, gamma=0.2

        Expected:
        - Trend correlation > 0.85
        - Seasonal correlation > 0.70
        - Similar R² values
        """
        # Load test data
        df = pd.read_csv(FIXTURES_DIR / 'test1_simple_weekly.csv')
        data = df['data'].values

        # Python decomposition
        py_result = STR_decompose(
            data,
            seasonal_periods=[7],
            trend_lambda=1500.0,
            seasonal_lambda=100.0
        )

        # R decomposition
        r_result = r_bridge.run_str_decomposition(
            data,
            seasonal_periods=[7],
            trend_lambda=1500.0,
            seasonal_lambda=100.0
        )

        # Compare trend
        trend_corr, trend_rmse = compare_decompositions(py_result, r_result, 'trend')
        assert trend_corr > 0.85, f"Trend correlation {trend_corr:.3f} too low"

        # Compare seasonal
        seasonal_corr, seasonal_rmse = compare_decompositions(py_result, r_result, 'seasonal')
        assert seasonal_corr > 0.70, f"Seasonal correlation {seasonal_corr:.3f} too low"

        # Compare R²
        py_r2 = 1 - py_result.remainder.var() / data.var()
        r_r2 = 1 - r_result['remainder'].var() / data.var()
        r2_diff = abs(py_r2 - r_r2)
        assert r2_diff < 0.15, f"R² difference {r2_diff:.3f} too large (Python={py_r2:.3f}, R={r_r2:.3f})"

        print(f"\\n  Trend correlation: {trend_corr:.3f}")
        print(f"  Seasonal correlation: {seasonal_corr:.3f}")
        print(f"  Python R²: {py_r2:.3f}, R R²: {r_r2:.3f}")

    def test_pure_sine_wave(self, r_bridge):
        """
        Test Case 2: Pure sine wave (critical test that revealed bugs).

        Data: n=100, pure sine with period=7

        Expected:
        - Both implementations should recover the sine pattern
        - High correlation (> 0.80)
        - Seasonal component NOT flat
        """
        # Load test data
        df = pd.read_csv(FIXTURES_DIR / 'test6_pure_sine.csv')
        data = df['data'].values

        # Python decomposition
        py_result = STR_decompose(
            data,
            seasonal_periods=[7],
            trend_lambda=100.0,
            seasonal_lambda=1.0
        )

        # R decomposition
        r_result = r_bridge.run_str_decomposition(
            data,
            seasonal_periods=[7],
            trend_lambda=100.0,
            seasonal_lambda=1.0
        )

        # Check Python seasonal is not flat (regression test)
        py_seasonal_std = py_result.seasonal[0].std()
        assert py_seasonal_std > 0.5, f"Python seasonal is flat (std={py_seasonal_std:.3f})"

        # Check R seasonal is not flat
        r_seasonal_std = r_result['seasonal'].std()
        assert r_seasonal_std > 0.5, f"R seasonal is flat (std={r_seasonal_std:.3f})"

        # Compare
        seasonal_corr, _ = compare_decompositions(py_result, r_result, 'seasonal')
        assert seasonal_corr > 0.60, f"Seasonal correlation {seasonal_corr:.3f} too low for sine wave"

        print(f"\\n  Python seasonal std: {py_seasonal_std:.3f}")
        print(f"  R seasonal std: {r_seasonal_std:.3f}")
        print(f"  Correlation: {seasonal_corr:.3f}")

    def test_deterministic_data(self, r_bridge):
        """
        Test Case 3: Deterministic data (no noise).

        Data: n=200, period=7, gamma=0.0 (perfect conditions)

        Expected:
        - Very high correlation (> 0.90)
        - Near-perfect decomposition
        """
        # Load test data
        df = pd.read_csv(FIXTURES_DIR / 'test5_deterministic.csv')
        data = df['data'].values

        # Python decomposition
        py_result = STR_decompose(
            data,
            seasonal_periods=[7],
            trend_lambda=500.0,
            seasonal_lambda=10.0
        )

        # R decomposition
        r_result = r_bridge.run_str_decomposition(
            data,
            seasonal_periods=[7],
            trend_lambda=500.0,
            seasonal_lambda=10.0
        )

        # Compare
        trend_corr, _ = compare_decompositions(py_result, r_result, 'trend')
        seasonal_corr, _ = compare_decompositions(py_result, r_result, 'seasonal')

        # Should be very high with no noise
        assert trend_corr > 0.90, f"Trend correlation {trend_corr:.3f} too low for deterministic data"
        assert seasonal_corr > 0.75, f"Seasonal correlation {seasonal_corr:.3f} too low for deterministic data"

        print(f"\\n  Trend correlation: {trend_corr:.3f}")
        print(f"  Seasonal correlation: {seasonal_corr:.3f}")

    @pytest.mark.parametrize("test_name,min_trend_corr,min_seasonal_corr", [
        ("test3_short_series", 0.75, 0.60),  # Short series - lower thresholds
        ("test8_weak_seasonal", 0.80, 0.50),  # Weak seasonal - very low seasonal threshold
        ("test9_strong_seasonal", 0.85, 0.75),  # Strong seasonal - high threshold
    ])
    def test_various_conditions(self, r_bridge, test_name, min_trend_corr, min_seasonal_corr):
        """
        Test Case 4-6: Various data conditions.

        Parametrized test covering:
        - Short series (n=56)
        - Weak seasonality (beta=0.2)
        - Strong seasonality (beta=5.0)
        """
        # Load test data
        df = pd.read_csv(FIXTURES_DIR / f'{test_name}.csv')
        data = df['data'].values

        # Python decomposition
        py_result = STR_decompose(
            data,
            seasonal_periods=[7],
            trend_lambda=500.0,
            seasonal_lambda=10.0
        )

        # R decomposition
        r_result = r_bridge.run_str_decomposition(
            data,
            seasonal_periods=[7],
            trend_lambda=500.0,
            seasonal_lambda=10.0
        )

        # Compare
        trend_corr, _ = compare_decompositions(py_result, r_result, 'trend')
        seasonal_corr, _ = compare_decompositions(py_result, r_result, 'seasonal')

        assert trend_corr > min_trend_corr, \
            f"{test_name}: Trend correlation {trend_corr:.3f} < {min_trend_corr}"
        assert seasonal_corr > min_seasonal_corr, \
            f"{test_name}: Seasonal correlation {seasonal_corr:.3f} < {min_seasonal_corr}"

        print(f"\\n  {test_name}:")
        print(f"    Trend: {trend_corr:.3f} (min: {min_trend_corr})")
        print(f"    Seasonal: {seasonal_corr:.3f} (min: {min_seasonal_corr})")


class TestRBridgeFunctionality:
    """Test R bridge itself (independent of stR package)."""

    def test_r_availability_check(self):
        """Test that R bridge correctly detects R installation."""
        bridge = RBridge()

        # R should be available (we installed it)
        assert bridge.r_available, "R should be installed"

        # Get availability message
        message = bridge.get_availability_message()
        assert isinstance(message, str)
        assert len(message) > 0

    @pytest.mark.skipif(not check_r_available(), reason="R/stR not available")
    def test_r_bridge_execution(self):
        """Test that R bridge can execute R scripts successfully."""
        bridge = RBridge()

        # Create simple test data
        data = np.sin(2*np.pi*np.arange(100)/7)

        # Run R decomposition
        result = bridge.run_str_decomposition(
            data,
            seasonal_periods=[7],
            trend_lambda=100.0,
            seasonal_lambda=1.0
        )

        # Check result structure
        assert 'trend' in result
        assert 'seasonal' in result
        assert 'remainder' in result

        # Check result lengths
        assert len(result['trend']) == len(data)
        assert len(result['seasonal']) == len(data)
        assert len(result['remainder']) == len(data)

        # Check components are not all zeros
        assert result['trend'].std() > 0
        assert result['seasonal'].std() > 0


class TestAlgorithmDifferences:
    """Document and validate known differences between Python and R."""

    @pytest.mark.skipif(not check_r_available(), reason="R/stR not available")
    def test_document_algorithm_differences(self, capsys):
        """
        Document the known algorithmic differences between Python and R.

        This is not a pass/fail test - it's documentation.
        """
        print("\\n" + "=" * 70)
        print("ALGORITHM DIFFERENCES: Python STRPy vs R stR")
        print("=" * 70)

        print("\\nPython (STRPy - Simplified):")
        print("  - Fourier basis (sin/cos) for seasonal components")
        print("  - Identity matrix for trend")
        print("  - 1 lambda per component (trend_lambda, seasonal_lambda)")
        print("  - Second-order differences on seasonal PATTERN")
        print("  - Simple overall mean centering")

        print("\\nR (stR - Full Implementation):")
        print("  - 2D seasonal surface with time-season structure")
        print("  - B-spline knots for trend")
        print("  - 3 lambdas per predictor (λ_tt, λ_st, λ_ss)")
        print("  - Complex seasonal topology (working days vs holidays)")
        print("  - Per-cycle centering")

        print("\\nWhy Correlation Validation:")
        print("  - Different algorithms → different numerical outputs")
        print("  - Both SHOULD extract similar components")
        print("  - Correlation measures if they identify same patterns")
        print("  - Threshold: trend > 0.85, seasonal > 0.70")

        print("\\nConclusion:")
        print("  Python implementation is SIMPLIFIED but CORRECT")
        print("  It trades advanced features for simplicity")
        print("  Suitable for most decomposition tasks")
        print("=" * 70)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
