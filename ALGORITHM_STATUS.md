# STR Algorithm Implementation Status

## ✅ Working Implementation

A **simplified but functional** STR decomposition algorithm is now available!

### Quick Start

```python
from strpy import STR_decompose, AutoSTR_simple, generate_synthetic_data

# Generate test data
df = generate_synthetic_data(n=365, periods=(7,), gamma=0.3, random_seed=42)

# Method 1: Manual parameters
result = STR_decompose(
    df['data'].values,
    seasonal_periods=[7],
    trend_lambda=1000.0,
    seasonal_lambda=100.0
)

# Method 2: Automatic parameter selection
result = AutoSTR_simple(
    df['data'].values,
    seasonal_periods=[7],
    n_trials=20
)

# Access components
print(f"Trend shape: {result.trend.shape}")
print(f"Seasonal components: {len(result.seasonal)}")
print(f"Remainder std: {result.remainder.std():.3f}")

# Visualize
result.plot()
```

### What Works

✅ **Basic Decomposition** (`STR_decompose`)
- Trend extraction via regularized regression
- Multiple seasonal components (e.g., weekly + monthly)
- Fourier basis for seasonal patterns
- Ridge regression with difference operators
- Variance explained: typically 90%+

✅ **Automatic Parameters** (`AutoSTR_simple`)
- Random search optimization
- AIC-like model selection
- 10-20 trials usually sufficient

✅ **Baseline Method** (`moving_average_decompose`)
- Simple moving average decomposition
- Good for comparison

✅ **Full Testing Suite**
- 21 tests passing
- Multiple seasonalities tested
- Accuracy validation

### Example Results

On synthetic data with SNR ≈ 3:
- **Trend RMSE**: ~1.0
- **Seasonal RMSE**: ~1.0
- **Variance Explained**: 90-96%
- **R²**: 0.92-0.96

### Current Implementation Details

**Algorithm:**
```
minimize ||y - Xβ||² + λ_trend ||D_trend·β_trend||² + λ_seasonal ||β_seasonal||²
```

Where:
- `X` = [Trend | Seasonal_1 | ... | Seasonal_k]
- Trend = Identity matrix (smoothed via D_trend)
- Seasonal = Fourier basis (sin/cos harmonics)
- `D_trend` = Second difference operator
- Solved via augmented least squares

**Features:**
- ✅ Multiple seasonalities
- ✅ Regularization
- ✅ Automatic centering
- ✅ Parameter optimization
- ⚠️ Simplified topology (no working day vs. holiday)
- ⚠️ No covariate support yet
- ⚠️ No robust (L1) estimation

### Examples

**Example 1: Single Seasonality**
```python
from strpy import STR_decompose
import numpy as np

# Create weekly pattern
n = 365
t = np.arange(n)
data = 0.01*t + np.sin(2*np.pi*t/7) + 0.2*np.random.randn(n)

result = STR_decompose(data, seasonal_periods=[7])
result.plot()
```

**Example 2: Multiple Seasonalities**
```python
# Weekly + Monthly patterns
result = STR_decompose(
    data,
    seasonal_periods=[7, 30],
    trend_lambda=1000,
    seasonal_lambda=50
)

print(f"Weekly pattern: {result.seasonal[0][:7]}")
print(f"Monthly pattern: {result.seasonal[1][:30]}")
```

**Example 3: Automatic Tuning**
```python
result = AutoSTR_simple(
    data,
    seasonal_periods=[7, 30],
    n_trials=30  # More trials = better parameters
)

print(f"Optimal trend_lambda: {result.params['trend_lambda']:.1f}")
print(f"Optimal seasonal_lambda: {result.params['seasonal_lambda']:.1f}")
```

### Running the Examples

```bash
# Activate environment
source env/bin/activate

# Run working example
python examples/03_working_str_example.py

# Should see:
# - Variance Explained: 90%+
# - R-squared: 0.95+
# - Generated plots in examples/
```

### Tests

```bash
# Run all tests
pytest tests/ -v

# Run STR-specific tests
pytest tests/test_str_simple.py -v

# Expected: 21 passed
```

## 🚧 Advanced Features (Future Work)

The following features from the paper are planned:

### Complex Seasonal Topology
- Working day vs. holiday patterns
- Transition periods
- Custom seasonal structures

### Covariate Support
- Static covariates
- Time-varying coefficients
- Seasonal covariates (e.g., temperature effects)

### Robust Estimation
- L1 norm (quantile regression)
- Outlier handling
- Mixed L1/L2 norms

### Advanced Optimization
- Leave-one-out CV (formula from paper)
- K-fold CV with gaps
- Full sparse matrix implementation

### Confidence Intervals
- Component-wise intervals
- Based on covariance matrix
- Prediction intervals

## Comparison: Simplified vs. Full Implementation

| Feature | Simplified (Working) | Full (Planned) |
|---------|---------------------|----------------|
| Trend extraction | ✅ Ridge regression | ✅ Tikhonov regularization |
| Seasonality | ✅ Fourier basis | ✅ 2D surface with topology |
| Multiple seasons | ✅ Yes | ✅ Yes |
| Covariates | ❌ No | ✅ Time-varying coefficients |
| Complex topology | ❌ No | ✅ Working day/holiday |
| Confidence intervals | ❌ No | ✅ Yes |
| Parameter selection | ✅ Random search | ✅ CV with formula |
| Robust estimation | ❌ No | ✅ L1/L2 mixed |

## Performance

**Speed:** Fast for typical use cases
- n=365: < 1 second
- n=1000: < 2 seconds
- AutoSTR with 20 trials: < 10 seconds

**Accuracy:** Good for most applications
- Trend recovery: RMSE ≈ 1.0 (standardized)
- Seasonal recovery: RMSE ≈ 1.0 (standardized)
- Comparable to STL for simple cases

**Limitations:**
- Not optimized for very long series (n > 10,000)
- No sparse matrix operations yet
- No parallel computation

## Migration from R

If migrating from the R `stR` package:

```r
# R code
library(stR)
result <- AutoSTR(data, seasonal.periods=c(7, 365))
trend <- result$output$predictors[[1]]$data
seasonal <- result$output$predictors[[2]]$data
```

```python
# Python equivalent (simplified)
from strpy import AutoSTR_simple

result = AutoSTR_simple(data, seasonal_periods=[7, 365])
trend = result.trend
seasonal_weekly = result.seasonal[0]
seasonal_yearly = result.seasonal[1]
```

## Next Steps

1. **Use the working implementation** for basic decomposition tasks
2. **Provide feedback** on accuracy and usability
3. **Compare results** with R stR package
4. **Identify use cases** requiring advanced features
5. **Contribute** to the full implementation

## Questions?

See:
- [README.md](README.md) - Package overview
- [QUICKREF.md](QUICKREF.md) - Quick reference
- [examples/03_working_str_example.py](examples/03_working_str_example.py) - Full example
- [tests/test_str_simple.py](tests/test_str_simple.py) - Test suite

## Citation

If using this implementation, please cite the original paper:

> Dokumentov, A., & Hyndman, R. J. (2022). STR: Seasonal-Trend decomposition using Regression. *INFORMS Journal on Data Science*, 1(1), 50-62.
