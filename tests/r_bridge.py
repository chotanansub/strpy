"""
Python-R bridge for STR validation tests.

This module provides utilities to:
1. Check if R and stR package are installed
2. Execute R scripts and exchange data via CSV
3. Compare Python and R decomposition results

Design: Uses subprocess + CSV for data exchange (no rpy2 dependency)
"""

import subprocess
import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


class RBridge:
    """Bridge to execute R scripts and exchange data."""

    def __init__(self):
        self.r_available = self._check_r_available()
        self.str_available = self._check_str_package() if self.r_available else False

    def _check_r_available(self) -> bool:
        """Check if R is installed and accessible."""
        try:
            result = subprocess.run(
                ['Rscript', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _check_str_package(self) -> bool:
        """Check if stR package is installed in R."""
        try:
            result = subprocess.run(
                ['Rscript', '-e', 'library(stR)'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def get_availability_message(self) -> str:
        """Get detailed message about R/stR availability."""
        if not self.r_available:
            return (
                "R is not installed. "
                "See docs/R_INSTALLATION.md for installation instructions."
            )
        elif not self.str_available:
            return (
                "R is installed but stR package is missing. "
                "Install with: Rscript -e 'install.packages(\"stR\")'"
            )
        else:
            return "R and stR package are available"

    def run_str_decomposition(
        self,
        data: np.ndarray,
        seasonal_periods: list,
        trend_lambda: float = 1500.0,
        seasonal_lambda: float = 100.0,
        script_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Run R STR decomposition and return results.

        Parameters
        ----------
        data : np.ndarray
            Time series data
        seasonal_periods : list
            List of seasonal periods
        trend_lambda : float
            Trend smoothing parameter
        seasonal_lambda : float
            Seasonal smoothing parameter
        script_path : str, optional
            Path to R script (default: tests/r_scripts/run_str.R)

        Returns
        -------
        dict
            Dictionary with keys: 'trend', 'seasonal', 'remainder'

        Raises
        ------
        RuntimeError
            If R or stR is not available, or execution fails
        """
        if not self.r_available or not self.str_available:
            raise RuntimeError(self.get_availability_message())

        # Default script path
        if script_path is None:
            script_path = Path(__file__).parent / 'r_scripts' / 'run_str.R'

        # Create temporary directory for data exchange
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save input data
            input_csv = tmpdir / 'input.csv'
            pd.DataFrame({'data': data}).to_csv(input_csv, index=False)

            # Prepare output path
            output_csv = tmpdir / 'output.csv'

            # Prepare R command
            cmd = [
                'Rscript',
                str(script_path),
                str(input_csv),
                str(output_csv),
                str(seasonal_periods[0]),  # Primary period
                str(trend_lambda),
                str(seasonal_lambda)
            ]

            # Execute R script
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        f"R script failed with code {result.returncode}\\n"
                        f"STDOUT: {result.stdout}\\n"
                        f"STDERR: {result.stderr}"
                    )

                # Load results
                if not output_csv.exists():
                    raise RuntimeError(
                        f"R script did not create output file\\n"
                        f"STDOUT: {result.stdout}\\n"
                        f"STDERR: {result.stderr}"
                    )

                df = pd.read_csv(output_csv)

                return {
                    'trend': df['trend'].values,
                    'seasonal': df['seasonal'].values,
                    'remainder': df['remainder'].values
                }

            except subprocess.TimeoutExpired:
                raise RuntimeError("R script execution timed out (>60s)")


def compare_decompositions(
    py_result,
    r_result: Dict[str, np.ndarray],
    component: str = 'trend'
) -> Tuple[float, float]:
    """
    Compare Python and R decomposition results.

    Parameters
    ----------
    py_result : STRResult
        Python STR decomposition result
    r_result : dict
        R decomposition result from run_str_decomposition()
    component : str
        Component to compare: 'trend', 'seasonal', or 'remainder'

    Returns
    -------
    correlation : float
        Pearson correlation coefficient
    rmse : float
        Root mean squared error
    """
    # Get Python component
    if component == 'trend':
        py_comp = py_result.trend
    elif component == 'seasonal':
        py_comp = py_result.seasonal[0]  # First seasonal component
    elif component == 'remainder':
        py_comp = py_result.remainder
    else:
        raise ValueError(f"Unknown component: {component}")

    # Get R component
    r_comp = r_result[component]

    # Ensure same length
    min_len = min(len(py_comp), len(r_comp))
    py_comp = py_comp[:min_len]
    r_comp = r_comp[:min_len]

    # Calculate correlation
    correlation = np.corrcoef(py_comp, r_comp)[0, 1]

    # Calculate RMSE
    rmse = np.sqrt(np.mean((py_comp - r_comp) ** 2))

    return correlation, rmse


def check_r_available() -> bool:
    """
    Check if R and stR package are available.

    Returns
    -------
    bool
        True if both R and stR package are installed
    """
    bridge = RBridge()
    return bridge.r_available and bridge.str_available


def get_r_version() -> Optional[str]:
    """
    Get R version string.

    Returns
    -------
    str or None
        R version string, or None if R is not available
    """
    try:
        result = subprocess.run(
            ['Rscript', '-e', 'cat(R.version.string)'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_str_version() -> Optional[str]:
    """
    Get stR package version.

    Returns
    -------
    str or None
        stR version string, or None if package is not available
    """
    try:
        result = subprocess.run(
            ['Rscript', '-e', 'cat(as.character(packageVersion("stR")))'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# Convenience function for pytest
def skip_if_r_unavailable():
    """
    Return pytest skip marker if R is not available.

    Usage:
        @skip_if_r_unavailable()
        def test_something():
            ...
    """
    import pytest
    bridge = RBridge()

    return pytest.mark.skipif(
        not (bridge.r_available and bridge.str_available),
        reason=bridge.get_availability_message()
    )


if __name__ == "__main__":
    # Self-test
    bridge = RBridge()

    print("=== R Bridge Status ===")
    print(f"R available: {bridge.r_available}")
    if bridge.r_available:
        print(f"R version: {get_r_version()}")

    print(f"stR package available: {bridge.str_available}")
    if bridge.str_available:
        print(f"stR version: {get_str_version()}")

    print(f"\\nMessage: {bridge.get_availability_message()}")

    if bridge.r_available and bridge.str_available:
        print("\\n✓ Ready for R validation tests!")
    else:
        print("\\n⚠️  R validation tests will be skipped")
        print("    See docs/R_INSTALLATION.md for setup instructions")
