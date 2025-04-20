import os
import sys
from src.data.data_loader import DataLoader
from src.visualization.plotter import DataPlotter
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Get the path to the data file
    data_path = "Gchart.csv"  # Update this path if needed
    
    # Initialize data loader and plotter
    data_loader = DataLoader(data_path)
    plotter = DataPlotter()
    
    try:
        # Load and preprocess data
        print("Loading data...")
        data = data_loader.load_data()
        
        print("Preprocessing data and calculating technical indicators...")
        data = data_loader.preprocess_data()
        
        # Create visualization directory if it doesn't exist
        os.makedirs("visualizations", exist_ok=True)
        
        # Generate and save basic plots
        print("Generating basic plots...")
        plotter.plot_ohlc(data, save_path="visualizations/ohlc_plot.png")
        plotter.plot_returns(data, save_path="visualizations/returns_plot.png")
        plotter.plot_volatility(data, save_path="visualizations/volatility_plot.png")
        plotter.plot_correlation(data, save_path="visualizations/correlation_plot.png")
        
        # Generate technical indicator plots
        print("Generating technical indicator plots...")
        
        # Plot Moving Averages
        ma_data = data[['Close', 'SMA_5', 'SMA_10', 'SMA_20']].copy()
        plt.figure(figsize=(12, 6))
        plt.plot(ma_data.index, ma_data['Close'], label='Close', alpha=0.5)
        plt.plot(ma_data.index, ma_data['SMA_5'], label='SMA 5')
        plt.plot(ma_data.index, ma_data['SMA_10'], label='SMA 10')
        plt.plot(ma_data.index, ma_data['SMA_20'], label='SMA 20')
        plt.title('Moving Averages')
        plt.legend()
        plt.savefig("visualizations/ma_plot.png")
        plt.close()
        
        # Plot RSI
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['RSI'])
        plt.axhline(y=70, color='r', linestyle='--')
        plt.axhline(y=30, color='g', linestyle='--')
        plt.title('Relative Strength Index (RSI)')
        plt.savefig("visualizations/rsi_plot.png")
        plt.close()
        
        # Plot MACD
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['MACD'], label='MACD')
        plt.plot(data.index, data['MACD_Signal'], label='Signal')
        plt.bar(data.index, data['MACD_Hist'], label='Histogram')
        plt.title('MACD')
        plt.legend()
        plt.savefig("visualizations/macd_plot.png")
        plt.close()
        
        # Plot Bollinger Bands
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], label='Close')
        plt.plot(data.index, data['BB_Upper'], label='Upper Band', alpha=0.5)
        plt.plot(data.index, data['BB_Middle'], label='Middle Band', alpha=0.5)
        plt.plot(data.index, data['BB_Lower'], label='Lower Band', alpha=0.5)
        plt.title('Bollinger Bands')
        plt.legend()
        plt.savefig("visualizations/bb_plot.png")
        plt.close()
        
        # Print data statistics
        print("\nData Statistics:")
        print(f"Number of trading days: {len(data)}")
        print(f"Average daily return: {data['Daily_Return'].mean():.4f}")
        print(f"Daily return standard deviation: {data['Daily_Return'].std():.4f}")
        print(f"Average daily volatility: {data['Daily_Volatility'].mean():.4f}")
        
        # Print technical indicator statistics
        print("\nTechnical Indicator Statistics:")
        print("RSI Statistics:")
        print(f"Mean: {data['RSI'].mean():.2f}")
        print(f"Std: {data['RSI'].std():.2f}")
        print(f"Min: {data['RSI'].min():.2f}")
        print(f"Max: {data['RSI'].max():.2f}")
        
        print("\nMACD Statistics:")
        print(f"Mean: {data['MACD'].mean():.4f}")
        print(f"Std: {data['MACD'].std():.4f}")
        
        # Prepare data for model training
        print("\nPreparing data for model training...")
        X_train, X_test, y_train, y_test = data_loader.get_train_test_split()
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 