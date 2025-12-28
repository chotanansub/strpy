"""
Working STR Decomposition Example

This example demonstrates the simplified but functional STR implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from strpy import generate_synthetic_data, STR_decompose, AutoSTR_simple
from strpy.simulations import rmse

# Set random seed
np.random.seed(42)

print("="*70)
print("STR Decomposition - Working Example")
print("="*70)

# Generate synthetic data
print("\n1. Generating synthetic time series...")
df = generate_synthetic_data(
    n=365,  # 1 year of daily data
    periods=(7,),  # Weekly seasonality
    alpha=1.0,
    beta=0.0,  # No yearly component for simplicity
    gamma=0.3,  # Moderate noise
    data_type="stochastic",
    random_seed=42
)

print(f"   Generated {len(df)} observations")
print(f"   True components available for comparison")

# Method 1: Manual parameter selection
print("\n2. STR Decomposition with manual parameters...")
result_manual = STR_decompose(
    df['data'].values,
    seasonal_periods=[7],
    trend_lambda=1000.0,
    seasonal_lambda=100.0
)

# Calculate errors
trend_rmse = rmse(df['trend'].values - result_manual.trend)
seasonal_rmse = rmse(df['seasonal_1'].values - result_manual.seasonal[0])

print(f"   Trend RMSE: {trend_rmse:.4f}")
print(f"   Seasonal RMSE: {seasonal_rmse:.4f}")
print(f"   Remainder std: {result_manual.remainder.std():.4f}")

# Method 2: Automatic parameter selection
print("\n3. Automatic parameter selection...")
result_auto = AutoSTR_simple(
    df['data'].values,
    seasonal_periods=[7],
    n_trials=20
)

# Calculate errors for auto result
trend_rmse_auto = rmse(df['trend'].values - result_auto.trend)
seasonal_rmse_auto = rmse(df['seasonal_1'].values - result_auto.seasonal[0])

print(f"\n   Auto STR Results:")
print(f"   Trend RMSE: {trend_rmse_auto:.4f}")
print(f"   Seasonal RMSE: {seasonal_rmse_auto:.4f}")
print(f"   Remainder std: {result_auto.remainder.std():.4f}")

# Visualization
print("\n4. Creating visualizations...")

fig, axes = plt.subplots(4, 2, figsize=(14, 12))

# Left column: True components
axes[0, 0].plot(df['data'], 'k-', linewidth=0.8)
axes[0, 0].set_ylabel('Data')
axes[0, 0].set_title('True Components', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(df['trend'], 'r-', linewidth=1.2)
axes[1, 0].set_ylabel('Trend')
axes[1, 0].grid(True, alpha=0.3)

axes[2, 0].plot(df['seasonal_1'][:56], 'g-', linewidth=1)  # First 8 weeks
axes[2, 0].set_ylabel('Seasonal')
axes[2, 0].grid(True, alpha=0.3)

axes[3, 0].plot(df['remainder'], 'gray', linewidth=0.5, alpha=0.7)
axes[3, 0].set_ylabel('Remainder')
axes[3, 0].set_xlabel('Time (days)')
axes[3, 0].grid(True, alpha=0.3)
axes[3, 0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Right column: Estimated components (Auto STR)
axes[0, 1].plot(result_auto.data, 'k-', linewidth=0.8)
axes[0, 1].set_ylabel('Data')
axes[0, 1].set_title('STR Estimated Components', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(result_auto.trend, 'r-', linewidth=1.2)
axes[1, 1].set_ylabel('Trend')
axes[1, 1].grid(True, alpha=0.3)

axes[2, 1].plot(result_auto.seasonal[0][:56], 'g-', linewidth=1)
axes[2, 1].set_ylabel('Seasonal')
axes[2, 1].grid(True, alpha=0.3)

axes[3, 1].plot(result_auto.remainder, 'gray', linewidth=0.5, alpha=0.7)
axes[3, 1].set_ylabel('Remainder')
axes[3, 1].set_xlabel('Time (days)')
axes[3, 1].grid(True, alpha=0.3)
axes[3, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('str_decomposition_comparison.png', dpi=150, bbox_inches='tight')
print("   Saved: str_decomposition_comparison.png")

# Detailed comparison plot
fig, axes = plt.subplots(3, 1, figsize=(14, 9))

# Trend comparison
axes[0].plot(df['trend'], 'r--', linewidth=1.5, alpha=0.5, label='True')
axes[0].plot(result_auto.trend, 'r-', linewidth=1, label='Estimated')
axes[0].set_ylabel('Trend', fontsize=12)
axes[0].set_title('Component Comparison: True vs Estimated', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Seasonal comparison (first 8 weeks)
axes[1].plot(df['seasonal_1'][:56], 'g--', linewidth=1.5, alpha=0.5, label='True')
axes[1].plot(result_auto.seasonal[0][:56], 'g-', linewidth=1, label='Estimated')
axes[1].set_ylabel('Seasonal', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Residuals
axes[2].plot(result_auto.remainder, 'gray', linewidth=0.5, alpha=0.7)
axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.8)
axes[2].set_ylabel('Remainder', fontsize=12)
axes[2].set_xlabel('Time (days)', fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('str_component_comparison.png', dpi=150, bbox_inches='tight')
print("   Saved: str_component_comparison.png")

# Summary statistics
print("\n5. Summary Statistics")
print("="*70)
print(f"{'Component':<20} {'True Std':<12} {'Est Std':<12} {'RMSE':<12}")
print("-"*70)
print(f"{'Trend':<20} {df['trend'].std():<12.4f} {result_auto.trend.std():<12.4f} {trend_rmse_auto:<12.4f}")
print(f"{'Seasonal':<20} {df['seasonal_1'].std():<12.4f} {result_auto.seasonal[0].std():<12.4f} {seasonal_rmse_auto:<12.4f}")
print(f"{'Remainder':<20} {df['remainder'].std():<12.4f} {result_auto.remainder.std():<12.4f} {'N/A':<12}")
print("="*70)

# Variance explained
total_var = df['data'].var()
explained_var = result_auto.fitted.var()
explained_pct = 100 * explained_var / total_var

print(f"\nVariance Explained: {explained_pct:.1f}%")
print(f"R-squared: {1 - result_auto.remainder.var() / df['data'].var():.4f}")

print("\n" + "="*70)
print("Example completed successfully!")
print("="*70)
print("\nNext steps:")
print("  - Try different lambda values")
print("  - Experiment with multiple seasonalities: periods=[7, 30]")
print("  - Compare with moving average baseline")
print("  - Use result.plot() for built-in visualization")
