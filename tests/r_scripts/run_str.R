#!/usr/bin/env Rscript
#
# R script to run STR decomposition for validation tests
#
# Usage:
#   Rscript run_str.R input.csv output.csv period trend_lambda seasonal_lambda
#
# Arguments:
#   input.csv       - CSV file with 'data' column
#   output.csv      - Output CSV file to write results
#   period          - Seasonal period (e.g., 7 for weekly)
#   trend_lambda    - Trend smoothing parameter (default: 1500)
#   seasonal_lambda - Seasonal smoothing parameter (default: 100)
#
# Output CSV columns:
#   - trend: Trend component
#   - seasonal: Seasonal component
#   - remainder: Remainder (residual) component
#

library(stR)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  cat("Usage: Rscript run_str.R <input.csv> <output.csv> <period> [trend_lambda] [seasonal_lambda]\\n")
  quit(status = 1)
}

input_file <- args[1]
output_file <- args[2]
period <- as.integer(args[3])
trend_lambda <- if (length(args) >= 4) as.numeric(args[4]) else 1500
seasonal_lambda <- if (length(args) >= 5) as.numeric(args[5]) else 100

cat("=== R STR Decomposition ===\\n")
cat("Input:  ", input_file, "\\n")
cat("Output: ", output_file, "\\n")
cat("Period: ", period, "\\n")
cat("Trend lambda:    ", trend_lambda, "\\n")
cat("Seasonal lambda: ", seasonal_lambda, "\\n")

# Load data
tryCatch({
  data_df <- read.csv(input_file)
  data <- data_df$data

  if (is.null(data) || length(data) == 0) {
    stop("Input CSV must have a 'data' column")
  }

  n <- length(data)
  cat("Data length: ", n, "\\n")

}, error = function(e) {
  cat("ERROR reading input file:\\n")
  cat(e$message, "\\n")
  quit(status = 1)
})

# Define predictors
# Following simplified approach comparable to Python implementation

# Trend predictor: Linear trend with smoothing
Trend <- list(
  data = rep(1, n),
  times = 1:n,
  seasons = rep(1, n),
  lambdas = c(trend_lambda, 0, 0)  # Only lambda_tt (time-time smoothing)
)

# Seasonal predictor: Seasonal pattern with cyclic smoothing
SeasonalComponent <- list(
  data = rep(1, n),
  times = 1:n,
  seasons = ((1:n - 1) %% period) + 1,  # 1-indexed seasonal cycle
  lambdas = c(0, 0, seasonal_lambda)     # Only lambda_ss (season-season smoothing)
)

cat("\\nRunning STR decomposition...\\n")

# Run STR
tryCatch({
  result <- STR(
    data,
    predictors = list(Trend, SeasonalComponent),
    confidence = NULL,  # No confidence intervals
    robust = FALSE      # Non-robust (matches Python default)
  )

  # Extract components
  trend <- result$output$predictors[[1]]$data
  seasonal <- result$output$predictors[[2]]$data
  remainder <- result$output$random$data

  # Verify lengths match
  if (length(trend) != n || length(seasonal) != n || length(remainder) != n) {
    stop(sprintf(
      "Component length mismatch: trend=%d, seasonal=%d, remainder=%d, expected=%d",
      length(trend), length(seasonal), length(remainder), n
    ))
  }

  # Save results
  output_df <- data.frame(
    trend = trend,
    seasonal = seasonal,
    remainder = remainder
  )

  write.csv(output_df, output_file, row.names = FALSE)

  cat("\\n✓ Decomposition complete\\n")
  cat("  Trend std:     ", sprintf("%.4f", sd(trend)), "\\n")
  cat("  Seasonal std:  ", sprintf("%.4f", sd(seasonal)), "\\n")
  cat("  Remainder std: ", sprintf("%.4f", sd(remainder)), "\\n")
  cat("  R²:            ", sprintf("%.4f", 1 - var(remainder)/var(data)), "\\n")
  cat("\\nResults saved to: ", output_file, "\\n")

}, error = function(e) {
  cat("ERROR during STR decomposition:\\n")
  cat(e$message, "\\n")
  quit(status = 1)
})

# Exit success
quit(status = 0)
