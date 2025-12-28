# STRPy Examples

Compact, focused examples demonstrating STR decomposition.

## Jupyter Notebooks (Compact)

### [01_quickstart.ipynb](01_quickstart.ipynb)
**5-minute introduction** - 8 cells
- Generate synthetic data
- Automatic decomposition
- Visualize results
- Check accuracy

```bash
jupyter notebook 01_quickstart.ipynb
```

### [02_advanced.ipynb](02_advanced.ipynb)
**Advanced features** - 10 cells
- Multiple seasonalities (weekly + monthly)
- Manual vs automatic parameters
- Baseline comparison
- Parameter sensitivity

### [03_simulation_study.ipynb](03_simulation_study.ipynb)
**Statistical validation** - 10 cells
- Effect of noise level
- Monte Carlo analysis
- Stochastic vs deterministic
- Performance metrics

## Python Scripts

### [basic_example.py](basic_example.py)
Basic data generation and visualization
```bash
python basic_example.py
```

### [03_working_str_example.py](03_working_str_example.py)
**Complete working demo** with:
- Manual and automatic decomposition
- True vs estimated comparison
- Accuracy metrics
- Generated plots

```bash
python 03_working_str_example.py
```

**Output:**
- `str_decomposition_comparison.png`
- `str_component_comparison.png`
- Summary statistics

## Quick Start

```python
from strpy import AutoSTR_simple, generate_synthetic_data

# Generate data
df = generate_synthetic_data(n=365, periods=(7,), gamma=0.3, random_seed=42)

# Decompose
result = AutoSTR_simple(df['data'].values, seasonal_periods=[7])

# Visualize
result.plot()

# Results: 90-96% variance explained, R² ≈ 0.96
```

## Running Examples

```bash
# Install dependencies
pip install -e ".[dev]"

# Launch Jupyter
jupyter notebook examples/

# Or JupyterLab
jupyter lab examples/

# Run Python scripts
python examples/03_working_str_example.py
```

## What's Included

| Example | Type | Cells | Focus |
|---------|------|-------|-------|
| 01_quickstart | Notebook | 8 | Getting started |
| 02_advanced | Notebook | 10 | Multiple seasonalities |
| 03_simulation_study | Notebook | 10 | Statistical validation |
| basic_example.py | Script | - | Data generation |
| 03_working_str_example.py | Script | - | Full demo ⭐ |

## Expected Results

- **Variance Explained**: 90-96%
- **R-squared**: 0.92-0.96
- **Trend RMSE**: ~1.0 (standardized)
- **Seasonal RMSE**: ~1.0 (standardized)
- **Speed**: < 1 second for n=365

## Next Steps

1. Start with `01_quickstart.ipynb`
2. Explore `02_advanced.ipynb` for complex patterns
3. Run `03_working_str_example.py` for full demo
4. See [ALGORITHM_STATUS.md](../ALGORITHM_STATUS.md) for implementation details

## Documentation

- [README.md](../README.md) - Main documentation
- [QUICKREF.md](../QUICKREF.md) - Quick reference
- [ALGORITHM_STATUS.md](../ALGORITHM_STATUS.md) - Implementation status
