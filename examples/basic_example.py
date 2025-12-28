"""
Basic example of using STRPy for time series decomposition.
"""

import numpy as np
import matplotlib.pyplot as plt
from strpy import generate_synthetic_data, STR

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data with weekly and yearly seasonality
print("Generating synthetic time series...")
df = generate_synthetic_data(
    n=1096,  # 3 years of daily data
    periods=(7, 365),  # weekly and yearly
    alpha=1.0,  # weekly component weight
    beta=1.0,  # yearly component weight
    gamma=0.25,  # noise weight
    data_type="stochastic",
    random_seed=42
)

print(f"Generated {len(df)} observations")
print(f"Data shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Plot the synthetic data
fig, axes = plt.subplots(5, 1, figsize=(12, 10))

axes[0].plot(df['data'], 'k-', linewidth=0.8)
axes[0].set_ylabel('Data')
axes[0].set_title('Synthetic Time Series Components')
axes[0].grid(True, alpha=0.3)

axes[1].plot(df['trend'], 'r-', linewidth=1.2)
axes[1].set_ylabel('True Trend')
axes[1].grid(True, alpha=0.3)

axes[2].plot(df['seasonal_1'], 'g-', linewidth=0.8)
axes[2].set_ylabel('True Weekly')
axes[2].grid(True, alpha=0.3)

axes[3].plot(df['seasonal_2'], 'b-', linewidth=0.8)
axes[3].set_ylabel('True Yearly')
axes[3].grid(True, alpha=0.3)

axes[4].plot(df['remainder'], 'gray', linewidth=0.5, alpha=0.7)
axes[4].set_ylabel('Remainder')
axes[4].set_xlabel('Time')
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('synthetic_data.png', dpi=150, bbox_inches='tight')
print("\nSaved plot to synthetic_data.png")

# Example STR decomposition (requires manual lambda tuning for now)
print("\n" + "="*60)
print("NOTE: Full STR implementation is in progress.")
print("The current version requires manual specification of lambdas.")
print("AutoSTR with automatic parameter selection is being developed.")
print("="*60)

# Example of how it will work (when implementation is complete):
"""
from strpy import AutoSTR

# Automatic decomposition
result = AutoSTR(
    df['data'].values,
    seasonal_periods=[7, 365],
    confidence=0.95
)

# Plot results
result.plot()
plt.savefig('str_decomposition.png', dpi=150, bbox_inches='tight')

# Access components
trend = result.trend
weekly_seasonal = result.seasonal[0]
yearly_seasonal = result.seasonal[1]
remainder = result.remainder

# Calculate RMSE
from strpy.simulations import rmse
print(f"Trend RMSE: {rmse(df['trend'].values - trend):.4f}")
print(f"Weekly RMSE: {rmse(df['seasonal_1'].values - weekly_seasonal):.4f}")
print(f"Yearly RMSE: {rmse(df['seasonal_2'].values - yearly_seasonal):.4f}")
"""

print("\nExample completed!")
