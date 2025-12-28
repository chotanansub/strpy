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

```python
import numpy as np
from strpy import STR, AutoSTR

# Generate sample data
n = 365 * 3
t = np.arange(n)
trend = 0.001 * t
seasonal = 10 * np.sin(2 * np.pi * t / 365)
noise = np.random.normal(0, 1, n)
y = trend + seasonal + noise

# Automatic STR decomposition
result = AutoSTR(y, seasonal_periods=[7, 365])

# Access components
trend_component = result.trend
seasonal_components = result.seasonal
remainder = result.remainder
```

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating:
- Basic decomposition with synthetic data
- Supermarket revenue analysis
- Electricity demand forecasting with temperature covariates

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
