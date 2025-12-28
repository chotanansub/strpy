"""
STRPy: Seasonal-Trend Decomposition using Regression

A Python implementation of the STR method for time series decomposition.
"""

__version__ = "0.1.0"
__author__ = "STRPy Contributors"

from .str import STR, AutoSTR
from .simulations import generate_synthetic_data
from .utils import STRResult

__all__ = [
    "STR",
    "AutoSTR",
    "generate_synthetic_data",
    "STRResult",
]
