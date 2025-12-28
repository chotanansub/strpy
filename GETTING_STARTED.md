# Getting Started with STRPy

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/chotanansub/strpy.git
cd strpy
```

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install the package
```bash
# For users
pip install -e .

# For developers (includes testing tools)
pip install -e ".[dev]"
```

## Quick Start

### Generate Synthetic Data

```python
import numpy as np
import matplotlib.pyplot as plt
from strpy import generate_synthetic_data

# Generate synthetic time series with multiple seasonalities
df = generate_synthetic_data(
    n=1096,              # 3 years of daily data
    periods=(7, 365),    # weekly and yearly seasonality
    alpha=1.0,           # weekly component weight
    beta=1.0,            # yearly component weight
    gamma=0.25,          # noise weight
    data_type="stochastic",
    random_seed=42
)

# View the data
print(df.head())
print(f"Columns: {df.columns.tolist()}")

# Plot components
fig, axes = plt.subplots(4, 1, figsize=(12, 8))
df['data'].plot(ax=axes[0], title='Data')
df['trend'].plot(ax=axes[1], title='Trend')
df['seasonal_1'].plot(ax=axes[2], title='Weekly Seasonal')
df['seasonal_2'].plot(ax=axes[3], title='Yearly Seasonal')
plt.tight_layout()
plt.show()
```

### STR Decomposition (Coming Soon)

```python
# Full implementation is being finalized
# This will work once complete:

from strpy import AutoSTR

# Automatic decomposition
result = AutoSTR(
    df['data'].values,
    seasonal_periods=[7, 365],
    confidence=0.95
)

# Access components
trend = result.trend
weekly = result.seasonal[0]
yearly = result.seasonal[1]
remainder = result.remainder

# Plot results with confidence intervals
result.plot()
plt.show()
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=strpy --cov-report=html

# View coverage report
open htmlcov/index.html  # On macOS
# or: xdg-open htmlcov/index.html  # On Linux
```

## Examples

Check the [`examples/`](examples/) directory for more examples:

- [`basic_example.py`](examples/basic_example.py) - Basic usage and data generation

## Project Structure

```
strpy/
├── src/strpy/          # Main package code
│   ├── str.py          # Core STR algorithm
│   ├── simulations.py  # Data generation
│   └── utils.py        # Utilities
├── tests/              # Unit tests
├── examples/           # Usage examples
├── legacy/             # Original R implementation
└── docs/               # Documentation (coming soon)
```

## Development Workflow

1. Make changes to the code
2. Run tests: `pytest tests/ -v`
3. Check code style: `black src/ tests/`
4. Run type checking: `mypy src/strpy/`
5. Update documentation as needed

## Current Status

### ✅ Working Features
- Data generation (synthetic time series)
- Utility functions (difference matrices, seasonal centering)
- Basic package structure
- Unit tests
- Documentation

### 🚧 In Development
- Full STR algorithm implementation
- Cross-validation for parameter selection
- Confidence interval calculations
- Complex seasonal topology support

### 📋 Planned
- Jupyter notebook tutorials
- Real-world examples
- Benchmarking suite
- Performance optimizations

## Getting Help

- **Documentation**: See [README.md](README.md) and [MIGRATION.md](MIGRATION.md)
- **Issues**: Report bugs or request features on GitHub
- **Legacy R Code**: Check the [`legacy/`](legacy/) directory for the original implementation

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## References

- **Paper**: Dokumentov & Hyndman (2022), "STR: Seasonal-Trend decomposition using Regression", INFORMS Journal on Data Science
- **R Package**: https://pkg.robjhyndman.com/stR/
- **Original Paper Code**: https://github.com/robjhyndman/STR_paper

## License

MIT License - see [LICENSE](LICENSE) for details
