import os
import logging
from src.data.data_loader import DataLoader
from src.models.trading_model import TradingModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    data_loader = DataLoader("Gchart.csv")
    data = data_loader.load_data()
    data = data_loader.preprocess_data()
    
    # Create sequences
    X, y = data_loader.create_sequences(data)
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Validation set shape: {X_val.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    
    # Initialize and train the model
    logger.info("Initializing and training the model...")
    model = TradingModel(input_shape=(X_train.shape[1], X_train.shape[2]))
    history = model.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # Plot training history
    os.makedirs("visualizations", exist_ok=True)
    model.plot_training_history(save_path="visualizations/training_history.png")
    
    # Evaluate on test set
    logger.info("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    
    # Evaluate trading signals
    metrics = model.evaluate_signals(y_test, y_pred)
    
    logger.info("\nTrading Signal Evaluation:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    
    logger.info("\nSignal Distribution:")
    logger.info(f"Total Signals: {metrics['total_signals']}")
    logger.info(f"Buy Signals: {metrics['buy_signals']} ({metrics['buy_signals']/metrics['total_signals']*100:.1f}%)")
    logger.info(f"Sell Signals: {metrics['sell_signals']} ({metrics['sell_signals']/metrics['total_signals']*100:.1f}%)")
    logger.info(f"Hold Signals: {metrics['hold_signals']}")
    
    logger.info("\nProfit Metrics:")
    logger.info(f"Total Return: {metrics['total_return']:.4f}")
    logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
    logger.info(f"Average Win: {metrics['avg_win']:.4f}")
    logger.info(f"Average Loss: {metrics['avg_loss']:.4f}")
    logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"True Positives: {metrics['true_positives']}")
    logger.info(f"True Negatives: {metrics['true_negatives']}")
    logger.info(f"False Positives: {metrics['false_positives']}")
    logger.info(f"False Negatives: {metrics['false_negatives']}")
    
    # Plot actual vs predicted signals
    plt.figure(figsize=(15, 6))
    plt.plot(y_test[:100], label='Actual Returns', alpha=0.7)
    plt.plot(y_pred[:100], label='Predicted Signals', alpha=0.7)
    plt.title('Actual Returns vs Predicted Signals (First 100 Samples)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("visualizations/signals_comparison.png")
    plt.close()
    
    # Plot signal distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred, bins=3, alpha=0.7, label='Predicted')
    plt.hist(y_test, bins=3, alpha=0.7, label='Actual')
    plt.title('Distribution of Trading Signals')
    plt.xlabel('Signal Type (-1: Sell, 0: Hold, 1: Buy)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig("visualizations/signal_distribution.png")
    plt.close()

if __name__ == "__main__":
    main() 