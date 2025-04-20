import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os
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

def plot_raw_data():
    """Main function to plot all raw indicators."""
    # Create directory for figures
    create_fig_directory()
    
    # Read the raw data
    df = pd.read_csv('Data.csv')
    
    # Get all indicators (excluding index and unnamed columns)
    all_indicators = [col for col in df.columns if col not in ['Index', 'Unnamed: 0']]
    
    # Create special price-volume plot
    print("Plotting price and volume data...")
    plot_price_volume(df)
    
    # Plot selected indicators overview
    print("Plotting selected indicators overview...")
    key_indicators = ['Close', 'RSI_12', 'MACD_12_26_9', 'BB_upper_20_2']
    plot_all_in_one(df, key_indicators)
    
    # Group indicators and plot each group
    groups = group_indicators(df)
    for group_name, indicators in groups.items():
        if indicators:  # Only plot if there are indicators in the group
            print(f"Plotting {group_name} raw data...")
            plot_indicator_group(df, group_name, indicators)

if __name__ == "__main__":
    # Set style for better visualization
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (15, 8)
    plt.rcParams['font.size'] = 10
    
    # Plot all raw data
    plot_raw_data() 