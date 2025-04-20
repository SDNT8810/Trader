import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os
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

def plot_normalized_data():
    """Main function to plot all normalized indicators."""
    # Create directory for figures
    create_fig_directory()
    
    # Read the normalized data
    df = pd.read_csv('NData.csv')
    
    # Get all indicators (excluding index and unnamed columns)
    all_indicators = [col for col in df.columns if col not in ['Index', 'Unnamed: 0']]
    
    # Plot all indicators in one figure
    print("Plotting all indicators in one figure...")
    plot_all_in_one(df, all_indicators)
    
    # Plot all indicators in separate rows
    print("Plotting all indicators in separate rows...")
    plot_all_rows(df, all_indicators)
    
    # Group indicators and plot each group
    groups = group_indicators(df)
    for group_name, indicators in groups.items():
        if indicators:  # Only plot if there are indicators in the group
            print(f"Plotting {group_name} indicators...")
            plot_indicator_group(df, group_name, indicators)

if __name__ == "__main__":
    # Set style for better visualization
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (15, 8)
    plt.rcParams['font.size'] = 10
    
    # Plot all normalized data
    plot_normalized_data()
