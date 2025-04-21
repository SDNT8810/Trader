import pandas as pd
import numpy as np
from typing import Optional

class CostFunction:
    """
    Calculate price changes for multiple time steps and their sum.
    
    Args:
        n_steps (int): Number of time steps to calculate changes for
        input_file (str): Path to input CSV file
        output_file (str): Path to save output CSV file
    """
    
    def __init__(self, n_steps: int = 10, input_file: str = 'Gchart.csv', output_file: str = 'Output.csv'):
        self.n_steps = n_steps
        self.input_file = input_file
        self.output_file = output_file
    
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file"""
        return pd.read_csv(self.input_file)
    
    def calculate_mean_price(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate mean price from OHLC data"""
        return (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
    
    def calculate_changes(self, mean_price: np.ndarray, time_data: pd.Series) -> pd.DataFrame:
        """
        Calculate price changes for n steps and their sum.
        For each step i, calculates P(t+i) - P(t).
        """
        length = len(mean_price)
        changes = np.zeros((length, self.n_steps))
        
        # Calculate changes for each step
        for i in range(1, self.n_steps + 1):
            # Shift prices i steps forward and subtract current prices
            future_prices = np.roll(mean_price, -i)
            # Calculate changes
            changes[:-i, i-1] = future_prices[:-i] - mean_price[:-i]
            # Pad the last i rows with NaN
            changes[-i:, i-1] = np.nan
        
        # Calculate total change (sum of all available changes for each time point)
        total_change = np.nansum(changes, axis=1)
        
        # Create DataFrame with column names
        columns = [f'Change_{i+1}' for i in range(self.n_steps)] + ['Total_Change']
        df = pd.DataFrame(np.column_stack([changes, total_change]), columns=columns)
        
        # Add time data
        df['Gmt time'] = time_data
        
        # Remove rows with NaN values
        df = df.dropna()
        
        return df
    
    def save_results(self, df: pd.DataFrame) -> None:
        """Save results to CSV file"""
        df.to_csv(self.output_file, index=False)
    
    def run(self) -> None:
        """Run the cost function calculation"""
        data = self.load_data()
        mean_price = self.calculate_mean_price(data)
        changes_df = self.calculate_changes(mean_price, data['Gmt time'])
        self.save_results(changes_df)
        print(f"Results saved to {self.output_file}")
        print(f"Output shape: {changes_df.shape}")
        print("\nSummary statistics:")
        print(changes_df.describe())

def main():
    """Run the cost function"""
    cost_function = CostFunction()
    cost_function.run()

if __name__ == "__main__":
    main() 