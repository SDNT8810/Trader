import pandas as pd
import numpy as np
import yaml
from typing import Optional
import os

def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load configuration from yaml file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class CostFunction:
    """
    Calculate mean prices for future time steps using OHLC data.
    
    Args:
        config_path (str): Path to configuration file
        input_file (str): Path to input CSV file (Data.csv)
        output_file (str): Path to save output CSV file (Output.csv)
    """
    
    def __init__(self, config_path: str = 'config/config.yaml', 
                 input_file: str = 'Data.csv', 
                 output_file: str = 'Output.csv'):
        self.config = load_config(config_path)
        self.prediction_window = self.config['model']['prediction_window']
        self.input_file = input_file
        self.output_file = output_file
    
    def load_data(self) -> pd.DataFrame:
        """Load OHLC data from CSV file"""
        return pd.read_csv(self.input_file)
    
    def calculate_mean_price(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate mean price from OHLC data"""
        return (data['High'] + data['Open'] + data['Close'] + data['Low']) / 4
    
    def calculate_future_means(self, mean_price: np.ndarray) -> pd.DataFrame:
        """
        Calculate mean prices for future time steps.
        For each time t, calculates mean prices from t+1 to t+prediction_window.
        """
        length = len(mean_price)
        future_means = np.zeros((length, self.prediction_window))
        
        # Calculate future means for each step
        for i in range(1, self.prediction_window + 1):
            # Shift prices i steps forward
            future_prices = np.roll(mean_price, -i)
            # Store future means
            future_means[:-i, i-1] = future_prices[:-i]
            # Pad the last i rows with NaN
            future_means[-i:, i-1] = np.nan
        
        # Create DataFrame with column names
        columns = ['index'] + [f'Mean_Price_t+{i}' for i in range(1, self.prediction_window + 1)]
        df = pd.DataFrame(np.column_stack([np.arange(length), future_means]), 
                         columns=columns)
        
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
        future_means_df = self.calculate_future_means(mean_price)
        self.save_results(future_means_df)
        print(f"Results saved to {self.output_file}")
        print(f"Output shape: {future_means_df.shape}")
        print("\nSummary statistics:")
        print(future_means_df.describe())

def main():
    """Run the cost function"""
    cost_function = CostFunction()
    cost_function.run()

if __name__ == "__main__":
    main() 