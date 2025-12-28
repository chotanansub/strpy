# Migration Guide: R to Python

This document describes the migration of the STR (Seasonal-Trend decomposition using Regression) implementation from R to Python.

## Overview

The original R project was located in the root directory and has been moved to [`legacy/`](legacy/). The new Python implementation is in [`src/strpy/`](src/strpy/).

## Project Structure

```
strpy/
├── legacy/                 # Original R implementation
│   ├── str_ijds.Rmd       # Paper source
│   ├── simulations.R      # Simulation functions
│   └── data/              # Example datasets
├── src/strpy/             # Python package
│   ├── __init__.py
│   ├── str.py             # Core STR implementation
│   ├── simulations.py     # Data generation
│   └── utils.py           # Utilities
├── tests/                 # Unit tests
├── examples/              # Usage examples
├── pyproject.toml         # Package configuration
└── README.md
```

## Key Differences

### R Implementation
- Uses the `stR` package from CRAN
- Main functions: `STR()`, `AutoSTR()`
- Data structures: R lists, data frames
- Matrix operations: Base R and sparse matrices

### Python Implementation
- Pure Python with NumPy, SciPy
- Main functions: `STR()`, `AutoSTR()`
- Data structures: NumPy arrays, pandas DataFrames
- Matrix operations: NumPy and SciPy sparse matrices

## Function Mapping

### Data Generation (simulations.R → simulations.py)

| R Function | Python Function | Notes |
|------------|----------------|-------|
| `get_trend()` | `generate_trend()` | Generates trend component |
| `get_seasonal()` | `generate_seasonal()` | Generates seasonal component |
| `get_msts_data()` | `generate_synthetic_data()` | Complete synthetic data |
| `compute_errors()` | `compute_decomposition_errors()` | Compare true vs estimated |

### Core STR (stR package → str.py)

| R Function | Python Class/Function | Notes |
|------------|----------------------|-------|
| `STR()` | `STR.fit()` | Main decomposition |
| `AutoSTR()` | `AutoSTR()` | Auto parameter selection |
| - | `STRResult` | Result container (dataclass) |

## Usage Comparison

### R Code
```r
library(stR)

# Generate data
data <- get_msts_data(type = "stochastic", l = 1096, gamma = 0.4)

# Decompose
result <- AutoSTR(data$data, seasonal.periods = c(7, 365))

# Access components
trend <- result$output$predictors[[1]]$data
seasonal_weekly <- result$output$predictors[[2]]$data
seasonal_yearly <- result$output$predictors[[3]]$data
remainder <- result$output$random$data
```

### Python Code
```python
from strpy import generate_synthetic_data, AutoSTR

# Generate data
df = generate_synthetic_data(
    n=1096,
    periods=(7, 365),
    gamma=0.4,
    data_type="stochastic"
)

# Decompose
result = AutoSTR(df['data'].values, seasonal_periods=[7, 365])

# Access components
trend = result.trend
seasonal_weekly = result.seasonal[0]
seasonal_yearly = result.seasonal[1]
remainder = result.remainder

# Plot
result.plot()
```

## Implementation Status

### ✅ Completed
- [x] Package structure and configuration
- [x] Data generation functions (simulations)
- [x] Core utility functions (difference matrices, etc.)
- [x] Basic STR algorithm structure
- [x] Result container with plotting
- [x] Unit tests for utilities and simulations
- [x] Documentation (README, docstrings)

### 🚧 In Progress
- [ ] Full STR matrix construction (complex seasonal topology)
- [ ] Cross-validation for parameter selection
- [ ] Confidence interval calculation
- [ ] Covariate handling (time-varying coefficients)

### 📋 Planned
- [ ] Jupyter notebook examples
- [ ] Comparison benchmarks (STR vs STL vs TBATS in Python)
- [ ] Integration tests with real data
- [ ] Performance optimization (sparse matrices)
- [ ] Documentation website

## Testing

Run tests with:
```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=strpy --cov-report=html
```

## Contributing

The Python implementation follows these principles:

1. **API Compatibility**: Keep similar interface to R version where possible
2. **Pythonic Code**: Use Python idioms and conventions
3. **Type Hints**: Add type annotations for better IDE support
4. **Documentation**: Comprehensive docstrings in NumPy format
5. **Testing**: Unit tests for all functions
6. **Performance**: Use vectorized NumPy operations

## References

- Original paper: Dokumentov & Hyndman (2022), INFORMS Journal on Data Science
- R package: https://pkg.robjhyndman.com/stR/
- Original code: https://github.com/robjhyndman/STR_paper

## Next Steps

1. Complete the full STR matrix construction for complex topology
2. Implement robust cross-validation
3. Add comprehensive integration tests
4. Create example notebooks with real datasets
5. Benchmark against R implementation
6. Publish to PyPI

## Notes

The current implementation is a work in progress. The core algorithm structure is in place, but some advanced features (complex seasonal topology, automatic parameter selection) require further development to match the full capabilities of the R version.

For questions or contributions, please open an issue on GitHub.
