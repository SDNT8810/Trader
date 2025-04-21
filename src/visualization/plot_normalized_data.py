import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
from typing import Dict, List
import numpy as np
import shutil

def create_fig_directory():
    """Create directory for saving figures if it doesn't exist."""
    fig_dir = 'src/Figs/Normalized'
    if os.path.exists(fig_dir):
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir)

def group_indicators(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Group indicators by their type."""
    # Exclude index and unnamed columns
    columns = [col for col in df.columns if col not in ['Index', 'Unnamed: 0']]
    
    groups = {
        'Price': ['Open', 'High', 'Low', 'Close'],
        'Volume': ['Volume'],
        'Moving Averages': [col for col in columns if any(ma in col for ma in ['SMA', 'EMA', 'DEMA', 'TEMA', 'WMA', 'TRIMA', 'KAMA'])],
        'Momentum': [col for col in columns if any(ind in col for ind in ['RSI', 'MFI', 'CCI', 'TRIX', 'ULTOSC'])],
        'Trend': [col for col in columns if any(ind in col for ind in ['ADX', 'PLUS_DI', 'MINUS_DI'])],
        'MACD': [col for col in columns if 'MACD' in col],
        'Bollinger Bands': [col for col in columns if 'BB_' in col],
        'Stochastic': [col for col in columns if 'STOCH' in col],
        'Other': [col for col in columns if not any(ind in col for ind in ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'EMA', 'DEMA', 'TEMA', 'WMA', 'TRIMA', 'KAMA', 'RSI', 'MFI', 'CCI', 'TRIX', 'ULTOSC', 'ADX', 'PLUS_DI', 'MINUS_DI', 'MACD', 'BB_', 'STOCH'])]
    }
    return groups

def plot_indicator_group(df: pd.DataFrame, group_name: str, indicators: List[str], figsize: tuple = (15, 8)):
    """Plot a group of related indicators and save to file."""
    plt.figure(figsize=figsize)
    
    # Create subplots based on the number of indicators
    n_indicators = len(indicators)
    n_rows = (n_indicators + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    fig.suptitle(f'{group_name} Indicators', fontsize=16)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, indicator in enumerate(indicators):
        row = idx // 2
        col = idx % 2
        
        # Plot the indicator
        ax = axes[row, col]
        ax.plot(df.index, df[indicator], label=indicator)
        
        # Set y-axis limits based on indicator type
        if any(x in indicator for x in ['CCI', 'MACD', 'MOM', 'ROC', 'Price_Change', 'APO', 'PPO', 'TRIX']):
            ax.set_ylim(-1.1, 1.1)
        elif 'WILLR' in indicator:
            ax.set_ylim(-1.1, 0.1)
        else:
            ax.set_ylim(-0.1, 1.1)
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_title(indicator)
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized Value')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Remove empty subplots if any
    for idx in range(len(indicators), n_rows * 2):
        row = idx // 2
        col = idx % 2
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the figure
    plt.savefig(f'src/Figs/Normalized/{group_name.lower().replace(" ", "_")}_indicators.png', dpi=600, bbox_inches='tight')
    plt.close()

def plot_all_in_one(df: pd.DataFrame, indicators: List[str]):
    """Plot all indicators in one figure."""
    plt.figure(figsize=(20, 12))
    
    # Create a color map
    colors = plt.cm.rainbow(np.linspace(0, 1, len(indicators)))
    
    # Plot each indicator
    for idx, (indicator, color) in enumerate(zip(indicators, colors)):
        plt.plot(df.index, df[indicator], label=indicator, color=color, alpha=0.7)
    
    plt.title('All Normalized Indicators', fontsize=16)
    plt.xlabel('Time')
    plt.ylabel('Normalized Value')
    plt.grid(True, alpha=0.3)
    
    # Add legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('src/Figs/Normalized/all_indicators.png', dpi=600, bbox_inches='tight')
    plt.close()

def plot_all_rows(df: pd.DataFrame, indicators: List[str]):
    """Plot all indicators in separate rows."""
    n_indicators = len(indicators)
    fig, axes = plt.subplots(n_indicators, 1, figsize=(20, 4*n_indicators))
    
    for idx, (indicator, ax) in enumerate(zip(indicators, axes)):
        ax.plot(df.index, df[indicator], label=indicator)
        
        # Set y-axis limits based on indicator type
        if any(x in indicator for x in ['CCI', 'MACD', 'MOM', 'ROC', 'Price_Change', 'APO', 'PPO', 'TRIX']):
            ax.set_ylim(-1.1, 1.1)
        elif 'WILLR' in indicator:
            ax.set_ylim(-1.1, 0.1)
        else:
            ax.set_ylim(-0.1, 1.1)
        
        ax.set_title(indicator)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Normalized Value')
        
        # Only show x-label for the last plot
        if idx != n_indicators - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig('src/Figs/Normalized/all_indicators_rows.png', dpi=600, bbox_inches='tight')
    plt.close()

class NormalizedDataPlotter:
    """
    Plot normalized technical indicators from NData.csv
    
    Features:
    - Configurable through config.yaml
    - Groups indicators by type
    - Creates subplots for each indicator group
    - Saves plots to Figs/ directory
    """
    
    def __init__(self, data_path: str = 'NData.csv', config_path: str = 'config/config.yaml'):
        """
        Initialize the plotter
        
        Parameters:
        -----------
        data_path : str
            Path to the normalized data CSV file
        config_path : str
            Path to the configuration file
        """
        self.data_path = data_path
        self.config = self._load_config(config_path)
        self.data = None
        self._validate_config()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _validate_config(self) -> None:
        """Validate the configuration structure."""
        if 'normalization' not in self.config:
            raise ValueError("Normalization configuration not found in config file")
        
        if 'indicators' not in self.config['normalization']:
            raise ValueError("Indicator configuration not found in config")
    
    def load_data(self) -> None:
        """Load the normalized data from CSV file."""
        self.data = pd.read_csv(self.data_path)
    
    def plot_indicators(self, save_path: str = 'src/Figs/Normalized/normalized_indicators.png') -> None:
        """
        Plot all normalized indicators
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot
        """
        if self.data is None:
            self.load_data()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Get indicator groups from config
        indicator_groups = self.config['normalization']['indicators']
        
        # Calculate number of subplots needed
        n_groups = len(indicator_groups)
        n_cols = 2
        n_rows = (n_groups + 1) // 2
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten()
        
        # Get normalization range
        norm_range = self.config['normalization']['methods']['min_max']['range']
        
        # Plot each indicator group
        for i, (group_name, indicators) in enumerate(indicator_groups.items()):
            ax = axes[i]
            
            # Plot each indicator in the group
            for indicator in indicators:
                if indicator in self.data.columns:
                    ax.plot(self.data[indicator], label=indicator)
            
            # Add normalization range lines
            ax.axhline(y=norm_range[0], color='r', linestyle='--', alpha=0.3)
            ax.axhline(y=norm_range[1], color='r', linestyle='--', alpha=0.3)
            
            ax.set_title(f'{group_name.upper()} Indicators (Normalized)')
            ax.legend()
            ax.grid(True)
        
        # Remove empty subplots
        for i in range(n_groups, len(axes)):
            fig.delaxes(axes[i])
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    def plot_price_with_indicators(self, save_path: str = 'src/Figs/Normalized/normalized_price_with_indicators.png') -> None:
        """
        Plot normalized price with selected indicators
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot
        """
        if self.data is None:
            self.load_data()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), height_ratios=[3, 1])
        
        # Get normalization range
        norm_range = self.config['normalization']['methods']['min_max']['range']
        
        # Plot price
        ax1.plot(self.data['Close'], label='Close Price', color='blue')
        ax1.set_ylabel('Normalized Price')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Add normalization range lines
        ax1.axhline(y=norm_range[0], color='r', linestyle='--', alpha=0.3)
        ax1.axhline(y=norm_range[1], color='r', linestyle='--', alpha=0.3)
        
        # Add selected indicators
        selected_indicators = [
            'ema_50', 'ema_200', 'bb_upper', 'bb_lower',
            'rsi_14', 'macd', 'macd_signal'
        ]
        
        for indicator in selected_indicators:
            if indicator in self.data.columns:
                if indicator in ['rsi_14']:
                    ax2.plot(self.data[indicator], label=indicator)
                else:
                    ax1.plot(self.data[indicator], label=indicator)
        
        ax1.legend()
        ax2.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

def main():
    """Main function to create plots"""
    # Set style for better visualization
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (15, 8)
    plt.rcParams['font.size'] = 10
    
    # Create plotter
    plotter = NormalizedDataPlotter()
    
    # Create plots
    plotter.plot_indicators()
    plotter.plot_price_with_indicators()

if __name__ == "__main__":
    main()
