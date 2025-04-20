import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

class DataPlotter:
    def __init__(self):
        """Initialize the DataPlotter with default style settings."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_ohlc(self, data: pd.DataFrame, title: str = "OHLC Data", save_path: Optional[str] = None):
        """
        Plot OHLC data with candlestick chart.
        
        Args:
            data (pd.DataFrame): DataFrame containing OHLC data
            title (str): Title for the plot
            save_path (str, optional): Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert dates to numerical format for plotting
        if 'Date' in data.columns:
            dates = pd.to_datetime(data['Date'])
            x_values = range(len(dates))
            # Set x-axis ticks and labels
            ax.set_xticks(x_values[::len(x_values)//10])  # Show ~10 date labels
            ax.set_xticklabels(dates.dt.strftime('%Y-%m-%d')[::len(x_values)//10], rotation=45)
        else:
            x_values = range(len(data))
        
        # Plot candlesticks
        for i, (idx, row) in enumerate(data.iterrows()):
            if row['Close'] >= row['Open']:
                color = 'green'
            else:
                color = 'red'
                
            # Plot the candle body
            ax.bar(x_values[i], row['Close'] - row['Open'], bottom=row['Open'], 
                  color=color, width=0.6)
            
            # Plot the wicks
            ax.plot([x_values[i], x_values[i]], [row['Low'], row['High']], color=color)
            
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        # plt.show()
        
    def plot_returns(self, data: pd.DataFrame, title: str = "Daily Returns", save_path: Optional[str] = None):
        """
        Plot daily returns distribution.
        
        Args:
            data (pd.DataFrame): DataFrame containing returns data
            title (str): Title for the plot
            save_path (str, optional): Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot returns time series
        ax1.plot(data.index, data['Daily_Return'])
        ax1.set_title('Daily Returns Time Series')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Return')
        
        # Plot returns distribution
        sns.histplot(data['Daily_Return'], kde=True, ax=ax2)
        ax2.set_title('Returns Distribution')
        ax2.set_xlabel('Return')
        ax2.set_ylabel('Frequency')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        # plt.show()
        
    def plot_volatility(self, data: pd.DataFrame, title: str = "Daily Volatility", save_path: Optional[str] = None):
        """
        Plot daily volatility.
        
        Args:
            data (pd.DataFrame): DataFrame containing volatility data
            title (str): Title for the plot
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Daily_Volatility'])
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        # plt.show()
        
    def plot_correlation(self, data: pd.DataFrame, title: str = "Feature Correlation", save_path: Optional[str] = None):
        """
        Plot correlation matrix of features.
        
        Args:
            data (pd.DataFrame): DataFrame containing features
            title (str): Title for the plot
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        # plt.show() 