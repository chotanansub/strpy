"""
Generate standard test datasets for R validation.

These datasets are used for comparing Python STRPy results against R stR package.
Each dataset tests different characteristics:
1. Simple weekly pattern (baseline)
2. Weekly + monthly (multiple seasonalities)
3. Short series (limited data)
4. High noise (robustness)
5. Deterministic (no noise)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from strpy import generate_synthetic_data


# Output directory
FIXTURES_DIR = Path(__file__).parent
FIXTURES_DIR.mkdir(exist_ok=True)


def save_dataset(df, name: str, description: str):
    """Save dataset to CSV with metadata."""
    output_path = FIXTURES_DIR / f"{name}.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path.name:30s} - {description}")


def generate_all_test_datasets():
    """Generate all standard test datasets."""
    print("\\nGenerating test datasets for R validation...")
    print("=" * 70)

    # Test Case 1: Simple weekly pattern (baseline)
    df1 = generate_synthetic_data(
        n=365,
        periods=(7,),
        alpha=1.0,
        beta=1.0,
        gamma=0.2,
        data_type="deterministic",
        random_seed=42
    )
    save_dataset(
        df1,
        "test1_simple_weekly",
        "n=365, period=7, gamma=0.2 (baseline)"
    )

    # Test Case 2: Weekly + monthly (multiple seasonalities)
    df2 = generate_synthetic_data(
        n=1096,  # 3 years
        periods=(7, 30),
        alpha=1.0,
        beta=1.0,
        gamma=0.3,
        data_type="deterministic",
        random_seed=123
    )
    save_dataset(
        df2,
        "test2_multiple_seasonalities",
        "n=1096, periods=[7,30], gamma=0.3"
    )

    # Test Case 3: Short series
    df3 = generate_synthetic_data(
        n=56,  # 8 weeks
        periods=(7,),
        alpha=1.0,
        beta=1.0,
        gamma=0.1,
        data_type="deterministic",
        random_seed=456
    )
    save_dataset(
        df3,
        "test3_short_series",
        "n=56, period=7, gamma=0.1 (short)"
    )

    # Test Case 4: High noise (robustness)
    df4 = generate_synthetic_data(
        n=365,
        periods=(7,),
        alpha=1.0,
        beta=1.0,
        gamma=0.8,
        data_type="deterministic",
        random_seed=789
    )
    save_dataset(
        df4,
        "test4_high_noise",
        "n=365, period=7, gamma=0.8 (high noise)"
    )

    # Test Case 5: Deterministic (no noise, perfect recovery expected)
    df5 = generate_synthetic_data(
        n=200,
        periods=(7,),
        alpha=1.0,
        beta=2.0,
        gamma=0.0,  # No noise
        data_type="deterministic",
        random_seed=999
    )
    save_dataset(
        df5,
        "test5_deterministic",
        "n=200, period=7, gamma=0.0 (no noise)"
    )

    # Test Case 6: Pure sine wave (critical test)
    n = 100
    t = np.arange(n)
    trend = 0.02 * t
    seasonal = 2 * np.sin(2*np.pi*t/7)
    noise = 0.1 * np.random.RandomState(42).randn(n)
    data = trend + seasonal + noise

    df6 = pd.DataFrame({
        'data': data,
        'trend': trend,
        'seasonal_1': seasonal,
        'noise': noise
    })
    save_dataset(
        df6,
        "test6_pure_sine",
        "n=100, pure sine, period=7 (bug test)"
    )

    # Test Case 7: Long series (performance test)
    df7 = generate_synthetic_data(
        n=2000,
        periods=(7,),
        alpha=1.0,
        beta=1.0,
        gamma=0.3,
        data_type="deterministic",
        random_seed=111
    )
    save_dataset(
        df7,
        "test7_long_series",
        "n=2000, period=7, gamma=0.3 (performance)"
    )

    # Test Case 8: Weak seasonality
    df8 = generate_synthetic_data(
        n=365,
        periods=(7,),
        alpha=1.0,
        beta=0.2,  # Weak seasonal
        gamma=0.3,
        data_type="deterministic",
        random_seed=222
    )
    save_dataset(
        df8,
        "test8_weak_seasonal",
        "n=365, period=7, beta=0.2 (weak seasonal)"
    )

    # Test Case 9: Strong seasonality
    df9 = generate_synthetic_data(
        n=365,
        periods=(7,),
        alpha=1.0,
        beta=5.0,  # Strong seasonal
        gamma=0.3,
        data_type="deterministic",
        random_seed=333
    )
    save_dataset(
        df9,
        "test9_strong_seasonal",
        "n=365, period=7, beta=5.0 (strong seasonal)"
    )

    # Test Case 10: Stochastic data
    df10 = generate_synthetic_data(
        n=365,
        periods=(7,),
        alpha=1.0,
        beta=1.0,
        gamma=0.3,
        data_type="stochastic",
        random_seed=444
    )
    save_dataset(
        df10,
        "test10_stochastic",
        "n=365, period=7, type=stochastic"
    )

    print("=" * 70)
    print(f"✓ Generated 10 test datasets in: {FIXTURES_DIR}")
    print("\\nDataset summary:")
    print("  Test 1-2:  Different scales and seasonalities")
    print("  Test 3:    Short series edge case")
    print("  Test 4:    High noise robustness")
    print("  Test 5-6:  Perfect/near-perfect conditions")
    print("  Test 7:    Long series performance")
    print("  Test 8-9:  Seasonal strength variation")
    print("  Test 10:   Stochastic vs deterministic")
    print("\\nThese datasets can be used with both Python and R implementations.")


if __name__ == "__main__":
    generate_all_test_datasets()
