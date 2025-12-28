# STRPy Quick Reference

## Installation

```bash
# Clone and setup
git clone https://github.com/chotanansub/strpy.git
cd strpy
python3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install package
pip install -e .          # User install
pip install -e ".[dev]"   # Developer install with testing tools
```

## Generate Synthetic Data

```python
from strpy import generate_synthetic_data

# Basic usage
df = generate_synthetic_data(
    n=1096,                    # Number of observations
    periods=(7, 365),          # Seasonal periods (weekly, yearly)
    alpha=1.0,                 # Weight for first seasonal
    beta=1.0,                  # Weight for second seasonal
    gamma=0.25,                # Noise level
    data_type="stochastic",    # "stochastic" or "deterministic"
    random_seed=42             # For reproducibility
)

# Returns DataFrame with columns:
# - data: Combined time series
# - trend: Trend component
# - seasonal_1: First seasonal component
# - seasonal_2: Second seasonal component
# - remainder: Noise component
```

## Individual Components

```python
from strpy.simulations import generate_trend, generate_seasonal

# Generate trend
trend = generate_trend(n=365, trend_type="stochastic")

# Generate seasonal pattern
seasonal = generate_seasonal(
    n=365,
    period=7,
    seasonal_type="deterministic",
    n_harmonics=5
)
```

## Utility Functions

```python
from strpy.utils import create_difference_matrix, center_seasonal

# Create difference matrix
D1 = create_difference_matrix(n=100, order=1, cyclic=False)  # First differences
D2 = create_difference_matrix(n=100, order=2, cyclic=True)   # Second differences (cyclic)

# Center seasonal component
centered = center_seasonal(seasonal_data, period=7)
```

## Error Metrics

```python
from strpy.simulations import rmse

# Calculate RMSE
error = rmse(true_values - estimated_values)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=strpy --cov-report=html

# Run specific test file
pytest tests/test_simulations.py -v
```

## Examples

```bash
# Run Python script
cd examples
python basic_example.py

# Launch Jupyter
jupyter notebook examples/
# or
jupyter lab examples/
```

## STR Decomposition (Coming Soon)

```python
from strpy import AutoSTR

# Will work once implementation is complete:
result = AutoSTR(
    data,
    seasonal_periods=[7, 365],
    confidence=0.95
)

# Access components
result.trend          # Trend component
result.seasonal[0]    # First seasonal
result.seasonal[1]    # Second seasonal
result.remainder      # Remainder/residuals

# Plot
result.plot()
```

## Common Workflows

### Simulation Study

```python
import numpy as np
from strpy import generate_synthetic_data
from strpy.simulations import rmse

# Generate multiple datasets
results = []
for seed in range(100):
    df = generate_synthetic_data(
        n=365,
        periods=(7,),
        gamma=0.3,
        random_seed=seed
    )
    results.append(df)

# Analyze variance contributions
for df in results[:5]:
    var_total = df['data'].var()
    var_trend = df['trend'].var()
    print(f"Trend contribution: {100*var_trend/var_total:.1f}%")
```

### Custom Data Generation

```python
from strpy.simulations import generate_trend, generate_seasonal
import numpy as np

n = 730  # 2 years daily

# Build custom series
trend = generate_trend(n, "stochastic")
weekly = generate_seasonal(n, period=7, seasonal_type="stochastic")
monthly = generate_seasonal(n, period=30, seasonal_type="deterministic")
noise = 0.2 * np.random.randn(n)

# Combine with custom weights
data = trend + 1.5*weekly + 0.8*monthly + noise
```

### Visualization

```python
import matplotlib.pyplot as plt

# Plot components
fig, axes = plt.subplots(5, 1, figsize=(12, 10))
df['data'].plot(ax=axes[0], title='Data')
df['trend'].plot(ax=axes[1], title='Trend')
df['seasonal_1'].plot(ax=axes[2], title='Weekly')
df['seasonal_2'].plot(ax=axes[3], title='Yearly')
df['remainder'].plot(ax=axes[4], title='Remainder')
plt.tight_layout()
plt.show()
```

## Project Structure

```
strpy/
├── src/strpy/              # Package source
│   ├── __init__.py
│   ├── str.py             # STR algorithm
│   ├── simulations.py     # Data generation
│   └── utils.py           # Utilities
├── tests/                  # Unit tests
│   ├── test_simulations.py
│   └── test_utils.py
├── examples/               # Usage examples
│   ├── basic_example.py
│   ├── 01_basic_usage.ipynb
│   └── 02_comparison_study.ipynb
├── legacy/                 # Original R code
├── pyproject.toml         # Package config
└── README.md              # Documentation
```

## Documentation

- [README.md](README.md) - Main documentation
- [GETTING_STARTED.md](GETTING_STARTED.md) - Quick start guide
- [MIGRATION.md](MIGRATION.md) - R to Python migration
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Complete overview
- [examples/README.md](examples/README.md) - Example guide

## Getting Help

```python
# In Python/IPython
help(generate_synthetic_data)
help(STR)

# View source
import inspect
print(inspect.getsource(generate_synthetic_data))
```

## Development

```bash
# Format code
black src/ tests/

# Type checking
mypy src/strpy/

# Lint
flake8 src/ tests/

# Build package
python -m build
```

## Troubleshooting

**Import Error:**
```bash
# Make sure package is installed
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/strpy/src"
```

**Test Failures:**
```bash
# Reinstall in editable mode
pip uninstall strpy
pip install -e .

# Clear cache
pytest --cache-clear
```

**Notebook Kernel Issues:**
```bash
# Install ipykernel
pip install ipykernel

# Register kernel
python -m ipykernel install --user --name=strpy
```

## Version Info

```python
import strpy
print(strpy.__version__)  # 0.1.0
```

---

For more details, see the full [README](README.md) or [documentation](GETTING_STARTED.md).
