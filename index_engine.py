import pandas as pd
import numpy as np

class BespokeIndexEngine:
    def __init__(self, prices_df: pd.DataFrame, initial_index_level: float = 1000.0):
        """
        Engine for constructing custom equity indices with dynamic Divisor maintenance.
        """
        self.prices = prices_df.sort_index()
        self.initial_level = initial_index_level
        self.tickers = prices_df.columns.tolist()
        
    def calculate_equal_weights(self) -> pd.Series:
        """Returns 1/N equal allocation across all constituents."""
        n = len(self.tickers)
        return pd.Series(1.0 / n, index=self.tickers)

    def calculate_inv_vol_weights(self, price_slice: pd.DataFrame, window: int = 30) -> pd.Series:
        """
        Smart Beta: Inverse Volatility weights calculated from trailing daily returns.
        Lower volatility stocks receive a higher index weight.
        """
        returns = price_slice.tail(window).pct_change().dropna()
        volatility = returns.std()
        
        # Protect against division by zero
        inv_vol = 1.0 / np.where(volatility == 0, np.nan, volatility)
        inv_vol_series = pd.Series(inv_vol, index=self.tickers).fillna(0)
        
        # Normalize weights to sum to 1.0 (100%)
        return inv_vol_series / inv_vol_series.sum()

    def run_rebalanced_index(self, schedule: str = "QE", scheme: str = "equal") -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulates the bespoke index over time, adjusting the Divisor on rebalance dates.
        
        Returns:
            index_df: DataFrame containing Index_Level and Divisor over time.
            weight_history_df: DataFrame where rows are rebalance dates and columns are ticker weights.
        """
        # Identify scheduled rebalance dates that exist in our trading calendar
        rebalance_dates = self.prices.resample(schedule).last().index
        rebalance_dates = rebalance_dates.intersection(self.prices.index)
        
        index_levels = []
        divisors = []
        weight_records = {}
        
        # Initialize Day 0 weights and divisor
        current_weights = self.calculate_equal_weights()
        first_date = self.prices.index[0]
        initial_portfolio_val = (self.prices.loc[first_date] * current_weights).sum()
        current_divisor = initial_portfolio_val / self.initial_level
        
        weight_records[first_date] = current_weights.to_dict()
        
        for date, row_prices in self.prices.iterrows():
            # 1. Check if today is a scheduled rebalance date
            if date in rebalance_dates and date != first_date:
                # Store the closing index level from yesterday/pre-rebalance
                old_index_level = (row_prices * current_weights).sum() / current_divisor
                
                # Calculate new target weights based on chosen methodology
                if scheme == "inv_vol":
                    hist_slice = self.prices.loc[:date]
                    current_weights = self.calculate_inv_vol_weights(hist_slice)
                else:
                    current_weights = self.calculate_equal_weights()
                
                # Record weights for Plotly visualization
                weight_records[date] = current_weights.to_dict()
                
                # Adjust Divisor: Divisor_new = (Sum of Price * New_Weight) / Old_Index_Level
                new_portfolio_val = (row_prices * current_weights).sum()
                current_divisor = new_portfolio_val / old_index_level
            
            # 2. Calculate daily Index Level
            portfolio_value = (row_prices * current_weights).sum()
            index_level = portfolio_value / current_divisor
            
            index_levels.append(index_level)
            divisors.append(current_divisor)
            
        index_df = pd.DataFrame(
            {"Index_Level": index_levels, "Divisor": divisors}, 
            index=self.prices.index
        )
        weight_history_df = pd.DataFrame.from_dict(weight_records, orient="index")
        
        return index_df, weight_history_df