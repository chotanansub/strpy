# STRPy Examples

This directory contains examples demonstrating the use of the STRPy package.

## Files

### Python Scripts

- **[basic_example.py](basic_example.py)** - Basic data generation and visualization
  - Generate synthetic time series
  - Visualize components
  - Save plots

### Jupyter Notebooks

- **[01_basic_usage.ipynb](01_basic_usage.ipynb)** - Comprehensive tutorial covering:
  - Data generation with different parameters
  - Component visualization
  - Stochastic vs deterministic comparison
  - Effect of noise levels
  - Variance contribution analysis

- **[02_comparison_study.ipynb](02_comparison_study.ipynb)** - Simulation study:
  - Monte Carlo simulations
  - Baseline decomposition methods
  - Error analysis across different configurations
  - Statistical comparisons (baseline + STR when complete)

## Running the Examples

### Python Scripts

```bash
# Activate virtual environment
source ../env/bin/activate  # On Windows: ..\env\Scripts\activate

# Run example
cd examples
python basic_example.py
```

### Jupyter Notebooks

```bash
# Install Jupyter if needed
pip install jupyter

# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Then open the desired notebook from the browser interface.

## Example Output

The `basic_example.py` script generates:
- `synthetic_data.png` - Visualization of generated time series components

## What's Next

When the full STR implementation is complete, these examples will be extended to include:

- STR decomposition with automatic parameter selection
- Comparison with STL and TBATS methods
- Real-world data examples:
  - Supermarket revenue analysis
  - Electricity demand with temperature covariates
  - Other time series from various domains

## Data

Generated synthetic data can be saved for later use:

```python
# In your notebook or script
df.to_csv('../data/my_synthetic_data.csv', index=False)
```

The [`data/`](../data/) directory is for storing example datasets (currently empty but ready for use).

## Customization

All examples can be customized by modifying parameters:

```python
from strpy import generate_synthetic_data

# Customize your data
df = generate_synthetic_data(
    n=730,                    # 2 years instead of 3
    periods=(7, 30, 365),     # Add monthly pattern
    alpha=1.5,                # Increase weekly weight
    beta=0.5,                 # Decrease yearly weight
    gamma=0.1,                # Lower noise
    data_type="deterministic", # Change generation type
    random_seed=12345         # Different seed
)
```

## Questions?

See the main [README](../README.md) or [GETTING_STARTED](../GETTING_STARTED.md) guide.
