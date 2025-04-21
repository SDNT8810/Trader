import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
from typing import Dict, List
import numpy as np
import shutil

def create_fig_directory():
    """Create directory for saving raw data figures if it doesn't exist."""
    fig_dir = 'src/Figs/RawFigs'
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

def set_y_limits(ax, indicator: str, data: pd.Series):
    """Set appropriate y-axis limits based on indicator type."""
    if any(x in indicator for x in ['RSI', 'MFI']):
        ax.set_ylim(-5, 105)  # RSI and MFI range from 0 to 100
    elif 'WILLR' in indicator:
        ax.set_ylim(-105, 5)  # WILLR ranges from -100 to 0
    elif 'CCI' in indicator:
        ax.set_ylim(-300, 300)  # CCI typical range
    elif any(x in indicator for x in ['ADX', 'DI']):
        ax.set_ylim(-5, 105)  # ADX and DI range from 0 to 100
    else:
        # For other indicators, use data range with some padding
        data_range = data.max() - data.min()
        ax.set_ylim(data.min() - 0.1 * data_range, data.max() + 0.1 * data_range)

def plot_indicator_group(df: pd.DataFrame, group_name: str, indicators: List[str], figsize: tuple = (15, 8)):
    """Plot a group of related indicators and save to file."""
    if not indicators:
        return
        
    # Create subplots based on the number of indicators
    n_indicators = len(indicators)
    n_rows = (n_indicators + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    fig.suptitle(f'{group_name} Raw Data', fontsize=16)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, indicator in enumerate(indicators):
        row = idx // 2
        col = idx % 2
        
        # Plot the indicator
        ax = axes[row, col]
        data = df[indicator].dropna()
        ax.plot(df.index, df[indicator], label=indicator)
        
        # Set appropriate y-axis limits
        set_y_limits(ax, indicator, data)
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_title(indicator)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        
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
    plt.savefig(f'src/Figs/RawFigs/{group_name.lower().replace(" ", "_")}_raw.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_price_volume(df: pd.DataFrame):
    """Create a special plot for price and volume data."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
    
    # Plot OHLC
    ax1.plot(df.index, df['High'], 'g-', alpha=0.3, label='High')
    ax1.plot(df.index, df['Low'], 'r-', alpha=0.3, label='Low')
    ax1.plot(df.index, df['Close'], 'b-', label='Close')
    ax1.fill_between(df.index, df['Low'], df['High'], color='gray', alpha=0.1)
    
    ax1.set_title('Price Data')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot Volume
    ax2.bar(df.index, df['Volume'], color='blue', alpha=0.3)
    ax2.set_title('Volume')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('src/Figs/RawFigs/price_volume.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_in_one(df: pd.DataFrame, indicators: List[str]):
    """Plot all indicators in one figure with multiple y-axes."""
    n_indicators = len(indicators)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_indicators))
    
    fig, host = plt.subplots(figsize=(20, 12))
    fig.subplots_adjust(right=0.75)
    
    axes = [host] + [host.twinx() for i in range(min(n_indicators - 1, 3))]
    
    # Spread additional axes
    for i, ax in enumerate(axes[2:], start=2):
        ax.spines["right"].set_position(("axes", 1 + (i-1)*0.1))
    
    curves = []
    for i, (indicator, color) in enumerate(zip(indicators[:4], colors[:4])):  # Limit to 4 indicators
        ax = axes[i]
        curve = ax.plot(df.index, df[indicator], label=indicator, color=color)
        curves.extend(curve)
        ax.set_ylabel(indicator)
        
        # Set y-axis limits
        data = df[indicator].dropna()
        data_range = data.max() - data.min()
        ax.set_ylim(data.min() - 0.1 * data_range, data.max() + 0.1 * data_range)
    
    host.set_xlabel("Time")
    
    # Add legend
    labels = [curve.get_label() for curve in curves]
    host.legend(curves, labels, loc='center left', bbox_to_anchor=(1.15, 0.5))
    
    plt.title('Selected Raw Indicators Overview', pad=20)
    plt.savefig('src/Figs/RawFigs/selected_indicators_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

class RawDataPlotter:
    """
    Plot raw technical indicators from Data.csv
    
    Features:
    - Configurable through config.yaml
    - Groups indicators by type
    - Creates subplots for each indicator group
    - Saves plots to Figs/ directory
    """
    
    def __init__(self, data_path: str = 'Data.csv', time_path: str = 'TimeData.csv', config_path: str = 'config/config.yaml'):
        """
        Initialize the plotter
        
        Parameters:
        -----------
        data_path : str
            Path to the raw data CSV file
        time_path : str
            Path to the time data CSV file
        config_path : str
            Path to the configuration file
        """
        self.data_path = data_path
        self.time_path = time_path
        self.config = self._load_config(config_path)
        self.data = None
        self.time = None
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
        """Load the raw data and time from CSV files."""
        self.data = pd.read_csv(self.data_path)
        self.time = pd.read_csv(self.time_path)
    
    def plot_indicators(self, save_path: str = 'src/Figs/RawData/raw_indicators.png') -> None:
        """
        Plot all raw indicators
        
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
        
        # Plot each indicator group
        for i, (group_name, indicators) in enumerate(indicator_groups.items()):
            ax = axes[i]
            
            # Plot each indicator in the group
            for indicator in indicators:
                if indicator in self.data.columns:
                    ax.plot(self.data[indicator], label=indicator)
            
            ax.set_title(f'{group_name.upper()} Indicators')
            ax.legend()
            ax.grid(True)
        
        # Remove empty subplots
        for i in range(n_groups, len(axes)):
            fig.delaxes(axes[i])
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    def plot_price_with_indicators(self, save_path: str = 'src/Figs/RawData/price_with_indicators.png') -> None:
        """
        Plot price with selected indicators
        
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
        
        # Plot price
        ax1.plot(self.data['Close'], label='Close Price', color='blue')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Plot volume
        ax2.plot(self.data['Volume'], label='Volume', color='purple')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
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
    plotter = RawDataPlotter()
    
    # Create plots
    plotter.plot_indicators()
    plotter.plot_price_with_indicators()

if __name__ == "__main__":
    main() 