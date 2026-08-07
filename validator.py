import pandas as pd
import numpy as np

def run_data_qa_pipeline(prices_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validates a structured time-series price DataFrame, logs anomalies,
    and imputes missing values using forward-fill methodology.
    
    Returns:
        cleaned_df: Cleaned price DataFrame ready for index calculation.
        audit_log: DataFrame summarizing QA checks per ticker.
    """
    audit_records = []
    cleaned_df = prices_df.copy()
    
    for ticker in cleaned_df.columns:
        series = cleaned_df[ticker]
        total_bars = len(series)
        
        # 1. Missing Value Detection
        missing_count = series.isna().sum()
        
        # 2. Stale Price Detection (price unchanged from previous day)
        stale_days = (series.diff() == 0).sum()
        
        # 3. Anomaly Detection: Flag unadjusted splits/crashes (Daily Return > ±20%)
        daily_returns = series.pct_change()
        extreme_jumps = (daily_returns.abs() > 0.20).sum()
        
        audit_records.append({
            "ticker": ticker,
            "total_bars": total_bars,
            "missing_bars": missing_count,
            "missing_pct": round((missing_count / total_bars) * 100, 2),
            "stale_days": stale_days,
            "extreme_movements_gt_20pct": extreme_jumps,
            "qa_status": "FLAGGED" if (missing_count > 5 or extreme_jumps > 0) else "PASSED"
        })
    
    # Imputation: S&P indices typically forward-fill last traded price for halted/holiday stocks
    cleaned_df = cleaned_df.ffill().bfill()
    
    audit_log = pd.DataFrame(audit_records)
    return cleaned_df, audit_log