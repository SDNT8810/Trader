import pandas as pd
import numpy as np

class CostFunction:
    """
    Calculate price changes for multiple time steps and their sum
    
    Features:
    - Calculates mean price (Open, High, Low, Close)
    - Computes price changes for n steps ahead
    - Calculates sum of all changes
    """
    
    def __init__(self, input_file: str = 'Gchart.csv', output_file: str = 'Output.csv', n_steps: int = 10):
        self.input_file = input_file
        self.output_file = output_file
        self.n_steps = n_steps
    
    def calculate_mean_price(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate mean price from OHLC data"""
        return data[['Open', 'High', 'Low', 'Close']].mean(axis=1).values
    
    def calculate_changes(self, mean_price: np.ndarray) -> np.ndarray:
        """Calculate price changes for each step: P(t+n) - P(t)"""
        changes = np.zeros((len(mean_price), self.n_steps))
        for n in range(1, self.n_steps + 1):
            changes[:, n-1] = np.roll(mean_price, -n) - mean_price
        return changes
    
    def calculate_sum(self, changes: np.ndarray) -> np.ndarray:
        """Calculate sum of all changes"""
        return np.sum(changes, axis=1)
    
    def process(self) -> None:
        """Process data and save results"""
        # Load data
        data = pd.read_csv(self.input_file)
        
        # Calculate mean price
        mean_price = self.calculate_mean_price(data)
        
        # Calculate changes for each step
        changes = self.calculate_changes(mean_price)
        
        # Calculate sum of changes
        total = self.calculate_sum(changes)
        
        # Create output DataFrame
        output = pd.DataFrame(changes, columns=[f'Change_{i}' for i in range(1, self.n_steps + 1)])
        output['Total_Change'] = total
        
        # Save results
        output.to_csv(self.output_file, index=False)
        print(f"Results saved to {self.output_file}")
        
        # Print summary
        print(f"\nShape: {output.shape}")
        print(f"Number of steps: {self.n_steps}")
        print("\nColumns:")
        for col in output.columns:
            values = output[col]
            print(f"{col}: [{values.min():.3f}, {values.max():.3f}]")

def main():
    """Run the cost function"""
    calculator = CostFunction(n_steps=10)  # Default to 10 steps
    calculator.process()

if __name__ == "__main__":
    main() 