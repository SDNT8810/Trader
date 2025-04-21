import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Optional
import yaml
import seaborn as sns
from scipy import stats

class ModelEvaluator:
    """
    Track and visualize model training progress and errors
    
    Features:
    - Tracks training and validation losses
    - Tracks prediction errors
    - Creates various visualization plots
    - Saves plots to Figs/Training directory
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config = self._load_config(config_path)
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.errors: List[Dict[str, np.ndarray]] = []  # Store errors for each epoch
        self.fig_dir = 'src/Figs/Training'
        
        # Create figure directory if it doesn't exist
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def update_losses(self, train_loss: float, val_loss: float) -> None:
        """Update training and validation losses"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
    
    def update_errors(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Update error tracking with new predictions and targets"""
        errors = {
            'absolute': np.abs(predictions - targets),
            'relative': np.abs((predictions - targets) / (targets + 1e-6)),  # Add small constant to avoid division by zero
            'squared': (predictions - targets) ** 2
        }
        self.errors.append(errors)
    
    def plot_progress(self, save_path: Optional[str] = None) -> None:
        """Plot training progress"""
        if not self.train_losses or not self.val_losses:
            print("No training data to plot")
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot losses
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        
        # Add labels and title
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_learning_curves(self, save_path: Optional[str] = None) -> None:
        """Plot learning curves with log scale"""
        if not self.train_losses or not self.val_losses:
            print("No training data to plot")
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot losses with log scale
        epochs = range(1, len(self.train_losses) + 1)
        plt.semilogy(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.semilogy(epochs, self.val_losses, 'r-', label='Validation Loss')
        
        # Add labels and title
        plt.title('Learning Curves (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.legend()
        plt.grid(True)
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_rolling_average(self, window_size: int = 10, save_path: Optional[str] = None) -> None:
        """Plot rolling average of losses"""
        if not self.train_losses or not self.val_losses:
            print("No training data to plot")
            return
            
        if len(self.train_losses) < window_size:
            print(f"Not enough data points for rolling average (need {window_size}, have {len(self.train_losses)})")
            return
        
        # Calculate rolling averages
        train_rolling = np.convolve(self.train_losses, np.ones(window_size)/window_size, mode='valid')
        val_rolling = np.convolve(self.val_losses, np.ones(window_size)/window_size, mode='valid')
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot rolling averages
        epochs = range(window_size, len(self.train_losses) + 1)
        plt.plot(epochs, train_rolling, 'b-', label=f'Training Loss ({window_size}-epoch avg)')
        plt.plot(epochs, val_rolling, 'r-', label=f'Validation Loss ({window_size}-epoch avg)')
        
        # Add labels and title
        plt.title(f'Rolling Average Loss (Window Size: {window_size})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_error_distribution(self, save_path: Optional[str] = None) -> None:
        """Plot distribution of prediction errors"""
        if not self.errors:
            print("No error data to plot")
            return
        
        # Get the latest errors
        latest_errors = self.errors[-1]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot absolute errors
        sns.histplot(latest_errors['absolute'], ax=axes[0], kde=True)
        axes[0].set_title('Absolute Error Distribution')
        axes[0].set_xlabel('Absolute Error')
        axes[0].set_ylabel('Frequency')
        
        # Plot relative errors
        sns.histplot(latest_errors['relative'], ax=axes[1], kde=True)
        axes[1].set_title('Relative Error Distribution')
        axes[1].set_xlabel('Relative Error')
        axes[1].set_ylabel('Frequency')
        
        # Plot squared errors
        sns.histplot(latest_errors['squared'], ax=axes[2], kde=True)
        axes[2].set_title('Squared Error Distribution')
        axes[2].set_xlabel('Squared Error')
        axes[2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_error_trends(self, save_path: Optional[str] = None) -> None:
        """Plot error trends over time"""
        if not self.errors:
            print("No error data to plot")
            return
        
        # Calculate mean errors for each epoch
        mean_abs_errors = [np.mean(err['absolute']) for err in self.errors]
        mean_rel_errors = [np.mean(err['relative']) for err in self.errors]
        mean_sq_errors = [np.mean(err['squared']) for err in self.errors]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot error trends
        epochs = range(1, len(self.errors) + 1)
        plt.plot(epochs, mean_abs_errors, 'b-', label='Mean Absolute Error')
        plt.plot(epochs, mean_rel_errors, 'g-', label='Mean Relative Error')
        plt.plot(epochs, mean_sq_errors, 'r-', label='Mean Squared Error')
        
        plt.title('Error Trends Over Training')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_error_correlation(self, save_path: Optional[str] = None) -> None:
        """Plot correlation between errors and price changes"""
        if not self.errors:
            print("No error data to plot")
            return
        
        # Get the latest errors and predictions
        latest_errors = self.errors[-1]
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot absolute error vs price change
        sns.scatterplot(x=latest_errors['absolute'], y=latest_errors['relative'], ax=axes[0])
        axes[0].set_title('Absolute vs Relative Error')
        axes[0].set_xlabel('Absolute Error')
        axes[0].set_ylabel('Relative Error')
        
        # Plot error distribution by price change magnitude
        sns.boxplot(x=np.digitize(latest_errors['absolute'], bins=5), y=latest_errors['relative'], ax=axes[1])
        axes[1].set_title('Error Distribution by Magnitude')
        axes[1].set_xlabel('Error Magnitude Bins')
        axes[1].set_ylabel('Relative Error')
        
        # Plot error autocorrelation
        sns.lineplot(x=range(len(latest_errors['absolute'])), y=latest_errors['absolute'], ax=axes[2])
        axes[2].set_title('Error Autocorrelation')
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Absolute Error')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_error_statistics(self, save_path: Optional[str] = None) -> None:
        """Plot error statistics and metrics"""
        if not self.errors:
            print("No error data to plot")
            return
        
        # Get the latest errors
        latest_errors = self.errors[-1]
        
        # Calculate statistics
        stats_data = {
            'Metric': ['Mean', 'Median', 'Std Dev', 'Max', 'Min', 'Skewness', 'Kurtosis'],
            'Absolute Error': [
                np.mean(latest_errors['absolute']),
                np.median(latest_errors['absolute']),
                np.std(latest_errors['absolute']),
                np.max(latest_errors['absolute']),
                np.min(latest_errors['absolute']),
                stats.skew(latest_errors['absolute']),
                stats.kurtosis(latest_errors['absolute'])
            ],
            'Relative Error': [
                np.mean(latest_errors['relative']),
                np.median(latest_errors['relative']),
                np.std(latest_errors['relative']),
                np.max(latest_errors['relative']),
                np.min(latest_errors['relative']),
                stats.skew(latest_errors['relative']),
                stats.kurtosis(latest_errors['relative'])
            ]
        }
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot bar chart of statistics
        x = np.arange(len(stats_data['Metric']))
        width = 0.35
        
        plt.bar(x - width/2, stats_data['Absolute Error'], width, label='Absolute Error')
        plt.bar(x + width/2, stats_data['Relative Error'], width, label='Relative Error')
        
        plt.title('Error Statistics')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.xticks(x, stats_data['Metric'], rotation=45)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_all_plots(self) -> None:
        """Save all available plots"""
        # Training progress plots
        self.plot_progress(os.path.join(self.fig_dir, 'training_progress.png'))
        self.plot_learning_curves(os.path.join(self.fig_dir, 'learning_curves.png'))
        self.plot_rolling_average(save_path=os.path.join(self.fig_dir, 'rolling_average.png'))
        
        # Error analysis plots
        self.plot_error_distribution(os.path.join(self.fig_dir, 'error_distribution.png'))
        self.plot_error_trends(os.path.join(self.fig_dir, 'error_trends.png'))
        self.plot_error_correlation(os.path.join(self.fig_dir, 'error_correlation.png'))
        self.plot_error_statistics(os.path.join(self.fig_dir, 'error_statistics.png'))

def main():
    """Example usage"""
    evaluator = ModelEvaluator()
    # Example data
    evaluator.train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
    evaluator.val_losses = [0.6, 0.5, 0.4, 0.3, 0.2]
    evaluator.save_all_plots()

if __name__ == '__main__':
    main() 