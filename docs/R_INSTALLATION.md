# R Installation for Validation Tests

## Overview

The STRPy validation tests can optionally compare Python results against the original R `stR` package implementation. However, R installation on macOS with Homebrew currently has compatibility issues.

## Known Issues

### Compiler Compatibility (macOS)

R 4.5.2 from Homebrew uses `-std=gnu23` which is not supported by Apple Clang 16:

```
error: invalid value 'gnu23' in '-std=gnu23'
```

This affects all R packages with C/C++ code, including:
- `stR` and its dependencies (forecast, quantreg, SparseM, Rcpp, etc.)
- Most CRAN packages

### Why This Happens

- R 4.5.2 was compiled with newer GNU compiler standards
- Apple Clang lags behind in C standard support
- Homebrew's R uses system compilers, which conflict

## Workaround Options

### Option 1: Official R Binary (Recommended)

Install R from the official CRAN website instead of Homebrew:

1. **Download R for macOS:**
   ```bash
   # Visit: https://cran.r-project.org/bin/macosx/
   # Download: R-4.4.2-arm64.pkg (or latest 4.4.x)
   ```

2. **Install stR package:**
   ```r
   install.packages("stR")
   library(stR)
   ```

3. **Verify installation:**
   ```bash
   Rscript -e 'library(stR); packageVersion("stR")'
   ```

### Option 2: Use Docker

Run R in a Docker container:

```dockerfile
# Create: docker/Dockerfile.r-validation
FROM r-base:4.4.2

RUN R -e 'install.packages("stR", repos="https://cloud.r-project.org")'

WORKDIR /workspace
COPY . /workspace

CMD ["Rscript", "tests/r_scripts/run_str.R"]
```

```bash
# Build and run
docker build -f docker/Dockerfile.r-validation -t strpy-r-validation .
docker run -v $(pwd)/tests/fixtures:/workspace/tests/fixtures strpy-r-validation
```

### Option 3: Conda/Mamba

Use conda-forge's R distribution:

```bash
# Create environment
conda create -n r-validation r-base=4.4 r-str

# Activate
conda activate r-validation

# Verify
Rscript -e 'library(stR)'
```

### Option 4: Wait for Homebrew Fix

Monitor Homebrew R formula updates:
```bash
brew update
brew info r  # Check for newer versions
```

## Running Validation Tests

### Without R Installed

Tests will automatically skip if R is not available:

```bash
pytest tests/test_r_comparison.py -v
# Output: SKIPPED - R not installed
```

### With R Installed

Tests will run if both R and stR package are detected:

```bash
# Check R availability
Rscript --version
Rscript -e 'library(stR)'

# Run validation tests
pytest tests/test_r_comparison.py -v
```

## Manual R Installation Steps

If you want to install R via Homebrew despite the issues:

```bash
# Install R
brew install r

# Create custom compiler config
mkdir -p ~/.R
cat > ~/.R/Makevars << 'EOF'
CC=clang
CXX=clang++
CFLAGS=-g -O2 -std=gnu11
CXXFLAGS=-g -O2 -std=gnu++17
EOF

# Try installing stR (may still fail)
Rscript -e 'install.packages("stR", repos="https://cloud.r-project.org")'
```

**Note:** This workaround may not work due to dependency chain issues.

## Alternative: Use Legacy R Code Directly

If R installation is too complex, you can:

1. Use the legacy R scripts in `legacy/` directory manually
2. Compare outputs offline
3. Contribute comparisons as regression test fixtures

Example:

```r
# In R console
source("legacy/simulations.R")
source("legacy/str_ijds.R")

# Generate test data
n <- 365
data <- generate_synthetic_data(n, periods=c(7), gamma=0.3)

# Run R STR
result_r <- STR_decompose(data$data, seasonal_periods=c(7))

# Save for comparison
write.csv(result_r, "tests/fixtures/r_output_test1.csv")
```

Then in Python:

```python
# Load R output
r_output = pd.read_csv("tests/fixtures/r_output_test1.csv")

# Run Python STR
py_result = STR_decompose(data, seasonal_periods=[7])

# Compare
np.corrcoef(py_result.trend, r_output['trend'])[0,1]
```

## Validation Test Design

### Why Correlation Instead of Exact Match

The Python implementation is **simplified** compared to R:

| Feature | R `stR` | Python `STRPy` |
|---------|---------|----------------|
| Seasonal basis | 2D surface with topology | Fourier basis (sin/cos) |
| Lambda values | 3 per predictor (λ_tt, λ_st, λ_ss) | 1 per component |
| Trend basis | B-splines with knots | Identity matrix |
| Regularization | Full Tikhonov with complex structure | Simplified difference matrices |

**Therefore:** We validate using **correlation** not exact numerical match:
- Trend correlation > 0.85
- Seasonal correlation > 0.70
- Similar variance decomposition

### Test Expectations

```python
# Good validation result:
✓ Trend correlation: 0.92 (> 0.85)
✓ Seasonal correlation: 0.78 (> 0.70)
✓ Variance ratio: 0.88 vs 0.90 (similar)

# Algorithm is working correctly!
```

## Future Work

1. **Pre-compiled R binaries**: Provide Docker images with R + stR pre-installed
2. **Fixture-based tests**: Include pre-computed R outputs in repository
3. **CI/CD integration**: Run R tests in GitHub Actions with official R Docker image
4. **Cross-platform support**: Test on Linux where compilation works better

## Getting Help

If you successfully install R and stR:
1. Share your setup in GitHub Issues
2. Contribute working installation scripts
3. Add your configuration to this documentation

If you encounter other issues:
- Check CRAN documentation: https://cran.r-project.org/doc/manuals/r-release/R-admin.html
- R-help mailing list: https://stat.ethz.ch/mailman/listinfo/r-help
- Stack Overflow: https://stackoverflow.com/questions/tagged/r

## References

- **stR package**: https://cran.r-project.org/package=stR
- **Original paper**: Dokumentov & Hyndman (2022), "STR: Seasonal-Trend decomposition using Regression"
- **R installation guide**: https://cran.r-project.org/doc/manuals/r-release/R-admin.html
- **Homebrew R formula**: https://github.com/Homebrew/homebrew-core/blob/master/Formula/r/r.rb
