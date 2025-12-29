#!/bin/bash
# Install R and stR package for validation tests

set -e  # Exit on error

echo "========================================"
echo "Installing R and stR package"
echo "========================================"

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Error: Homebrew not found. Please install Homebrew first:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

# Install R
echo ""
echo "Step 1: Installing R via Homebrew..."
if command -v R &> /dev/null; then
    echo "✓ R already installed: $(R --version | head -n1)"
else
    brew install r
    echo "✓ R installed successfully"
fi

# Verify R installation
echo ""
echo "Step 2: Verifying R installation..."
R_VERSION=$(R --version | head -n1)
echo "✓ R version: $R_VERSION"

# Install stR package from CRAN
echo ""
echo "Step 3: Installing stR package from CRAN..."
Rscript -e 'if (!require("stR", quietly = TRUE)) { install.packages("stR", repos="https://cloud.r-project.org", quiet = TRUE) }; library(stR); cat("✓ stR package version:", as.character(packageVersion("stR")), "\n")'

# Verify stR installation
echo ""
echo "Step 4: Verifying stR package..."
Rscript -e 'library(stR); cat("✓ stR package loaded successfully\n")'

# Install additional dependencies for tests
echo ""
echo "Step 5: Installing R dependencies for tests..."
Rscript -e 'packages <- c("readr", "dplyr"); new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]; if(length(new_packages)) install.packages(new_packages, repos="https://cloud.r-project.org", quiet = TRUE); lapply(packages, library, character.only = TRUE, quietly = TRUE); cat("✓ Dependencies installed: readr, dplyr\n")'

echo ""
echo "========================================"
echo "✓ Installation complete!"
echo "========================================"
echo ""
echo "R and stR package are ready for validation tests."
echo "You can now run: pytest tests/test_r_comparison.py"
