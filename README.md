# STRPy: Seasonal-Trend Decomposition using Regression

Python implementation of the STR (Seasonal-Trend decomposition using Regression) method for time series decomposition.

## Overview

STRPy provides a flexible framework for decomposing seasonal time series data into:
- **Trend component**: Smoothly changing underlying mean
- **Seasonal components**: Multiple seasonal patterns with possibly complex topology
- **Covariates**: External predictors with time-varying coefficients
- **Remainder**: Residual noise and idiosyncratic patterns

## Key Features

- ✅ Multiple seasonal components (e.g., daily, weekly, yearly patterns)
- ✅ Complex seasonal topology (e.g., working day vs. holiday patterns)
- ✅ Time-varying coefficients for covariates
- ✅ Automatic parameter selection via cross-validation
- ✅ Confidence intervals for all components
- ✅ Robust to missing data
- ✅ Handles fractional seasonal periods

## Installation

### From source
```bash
git clone https://github.com/chotanansub/strpy.git
cd strpy
pip install -e .
```

### Development installation
```bash
pip install -e ".[dev]"
```

## Quick Start

> **✨ NEW: Working implementation now available!** See [ALGORITHM_STATUS.md](ALGORITHM_STATUS.md) for details.

```python
from strpy import STR_decompose, AutoSTR_simple, generate_synthetic_data

# Generate sample data
df = generate_synthetic_data(
    n=365,
    periods=(7,),  # Weekly seasonality
    gamma=0.3,
    random_seed=42
)

# Method 1: Automatic parameter selection (recommended)
result = AutoSTR_simple(
    df['data'].values,
    seasonal_periods=[7],
    n_trials=20
)

# Method 2: Manual parameters
result = STR_decompose(
    df['data'].values,
    seasonal_periods=[7],
    trend_lambda=1000.0,
    seasonal_lambda=100.0
)

# Access components
print(f"Trend: {result.trend.shape}")
print(f"Seasonal: {len(result.seasonal)} components")
print(f"Variance explained: {100*(1-result.remainder.var()/df['data'].var()):.1f}%")

# Visualize
result.plot()
```

## Examples

See the `examples/` directory for:

**Python Scripts:**
- [basic_example.py](examples/basic_example.py) - Data generation and visualization
- [03_working_str_example.py](examples/03_working_str_example.py) - **Working STR decomposition** ✨

**Jupyter Notebooks:**
- [01_basic_usage.ipynb](examples/01_basic_usage.ipynb) - Comprehensive data generation tutorial (24 cells)
- [02_comparison_study.ipynb](examples/02_comparison_study.ipynb) - Simulation study and method comparison (19 cells)

Run examples:
```bash
# Python scripts
python examples/03_working_str_example.py

# Jupyter notebooks
jupyter notebook examples/
```

## Method

STR recasts time series decomposition as a regularized regression problem:

```
y_t = T_t + Σ S^(i)_t + Σ φ_{p,t} z_{t,p} + R_t
```

where:
- `T_t` is a smooth trend
- `S^(i)_t` are seasonal components with complex topology
- `z_{p,t}` are covariates with time-varying coefficients `φ_{p,t}`
- `R_t` is the remainder

The model uses regularization via difference operators to ensure smooth components, similar to ridge regression.

## References

This implementation is based on:

> Dokumentov, A., & Hyndman, R. J. (2022). STR: Seasonal-Trend decomposition using Regression. *INFORMS Journal on Data Science*, 1(1), 50-62.

Original R implementation: [stR package](https://pkg.robjhyndman.com/stR/)

## License

MIT License

## Citation

If you use this package in your research, please cite:

```bibtex
@article{dokumentov2022str,
  title={STR: Seasonal-Trend decomposition using Regression},
  author={Dokumentov, Alexander and Hyndman, Rob J},
  journal={INFORMS Journal on Data Science},
  volume={1},
  number={1},
  pages={50--62},
  year={2022},
  publisher={INFORMS}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Legacy Code

The original R-based research code is preserved in the `legacy/` directory.
