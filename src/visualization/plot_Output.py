import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class OutputPlotter:
    """
    Plot cost function output data
    
    Features:
    - Creates subplots for each time step
    - Plots total change
    - Saves plots to Figs/Reward directory
    """
    
    def __init__(self, input_file: str = 'Output.csv', fig_dir: str = 'src/Figs/Reward'):
        self.input_file = input_file
        self.fig_dir = fig_dir
        self.data = None
        
        # Create figure directory if it doesn't exist
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
    
    def load_data(self) -> None:
        """Load output data"""
        self.data = pd.read_csv(self.input_file)
        # Convert time column to datetime with European format (DD.MM.YYYY)
        self.data['Gmt time'] = pd.to_datetime(self.data['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
    
    def plot_changes(self) -> None:
        """Plot individual changes and total against time"""
        if self.data is None:
            self.load_data()
        
        # Get change columns and total
        change_cols = [col for col in self.data.columns if col.startswith('Change_')]
        n_changes = len(change_cols)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 4)
        
        # Plot individual changes
        for i, col in enumerate(change_cols):
            row = i // 4
            col_idx = i % 4
            ax = fig.add_subplot(gs[row, col_idx])
            
            sns.lineplot(x='Gmt time', y=col, data=self.data, ax=ax)
            ax.set_title(f'Step {i+1}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price Change')
            plt.xticks(rotation=45)
        
        # Plot total change
        ax = fig.add_subplot(gs[2, :2])
        sns.lineplot(x='Gmt time', y='Total_Change', data=self.data, ax=ax, color='red')
        ax.set_title('Total Change')
        ax.set_xlabel('Time')
        ax.set_ylabel('Total Price Change')
        plt.xticks(rotation=45)
        
        # Plot distribution of total change
        ax = fig.add_subplot(gs[2, 2:])
        sns.histplot(data=self.data['Total_Change'], ax=ax, color='red')
        ax.set_title('Total Change Distribution')
        ax.set_xlabel('Total Price Change')
        ax.set_ylabel('Count')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, 'price_changes.png'))
        plt.close()
        
        print(f"Plot saved to {os.path.join(self.fig_dir, 'price_changes.png')}")
    
    def plot_heatmap(self) -> None:
        """Plot correlation heatmap of changes"""
        if self.data is None:
            self.load_data()
        
        # Create correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation between Price Changes')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, 'correlation_heatmap.png'))
        plt.close()
        
        print(f"Heatmap saved to {os.path.join(self.fig_dir, 'correlation_heatmap.png')}")
    
    def plot_all(self) -> None:
        """Generate all plots"""
        self.plot_changes()
        self.plot_heatmap()

def main():
    """Run the visualization"""
    plotter = OutputPlotter()
    plotter.plot_all()

if __name__ == "__main__":
    main() 