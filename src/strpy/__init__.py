"""
STRPy: Seasonal-Trend Decomposition using Regression

A Python implementation of the STR method for time series decomposition.
"""

__version__ = "0.1.0"
__author__ = "STRPy Contributors"

# Working simplified implementation
from .str_simple import STR_decompose, AutoSTR_simple, moving_average_decompose

# Full implementation (in development)
from .str import STR, AutoSTR

from .simulations import generate_synthetic_data
from .utils import STRResult

__all__ = [
    # Simplified working functions
    "STR_decompose",
    "AutoSTR_simple",
    "moving_average_decompose",
    # Full implementation (in progress)
    "STR",
    "AutoSTR",
    # Utilities
    "generate_synthetic_data",
    "STRResult",
]
