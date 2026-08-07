import pandas as pd
import numpy as np
from validator import run_data_qa_pipeline
from index_engine import BespokeIndexEngine

# 1. Generate 1 Year of Synthetic Daily Stock Data (3 Tickers)
np.random.seed(42)
dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="B")
raw_data = {
    "TICKER_A": 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, len(dates))),
    "TICKER_B": 250 * np.cumprod(1 + np.random.normal(0.0003, 0.010, len(dates))),
    "TICKER_C": 50  * np.cumprod(1 + np.random.normal(0.0007, 0.025, len(dates))),
}
df_prices = pd.DataFrame(raw_data, index=dates)

# Inject synthetic anomalies to test our Data QA script
df_prices.iloc[10:13, 0] = np.nan            # Simulate 3 missing bars for TICKER_A
df_prices.iloc[45, 2] *= 1.35                # Simulate a +35% unadjusted jump for TICKER_C

# 2. Run Data QA Pipeline
cleaned_prices, audit_log = run_data_qa_pipeline(df_prices)
print("=== DATA VALIDATION AUDIT LOG ===")
print(audit_log.to_string(index=False))
print("\n")

# 3. Run Bespoke Index Calculations
engine = BespokeIndexEngine(cleaned_prices, initial_index_level=1000.0)

# Build Equal-Weight Index (Quarterly Rebalance)
ew_index = engine.run_rebalanced_index(schedule="QE", scheme="equal")

# Build Inverse-Volatility Smart Beta Index (Quarterly Rebalance)
iv_index = engine.run_rebalanced_index(schedule="QE", scheme="inv_vol")

# 4. Compare Performance Outputs
results = pd.DataFrame({
    "Equal_Weight_Level": ew_index["Index_Level"],
    "Smart_Beta_Level": iv_index["Index_Level"],
    "EW_Divisor": ew_index["Divisor"]
})

print("=== INDEX PERFORMANCE SNAPSHOT (First 5 Days) ===")
print(results.head().round(2))
print("\n=== INDEX PERFORMANCE SNAPSHOT (Last 5 Days) ===")
print(results.tail().round(2))