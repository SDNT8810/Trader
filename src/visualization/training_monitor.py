import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import os
from datetime import datetime
import psutil
import torch
from typing import Dict, List, Optional
import json
import yaml

class TrainingMonitor:
    def __init__(self, config_path: str = 'config/config.yaml', save_dir: str = 'src/Figs/Training'):
        """Initialize the training monitor"""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize metrics storage
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': [],
            'gradient_norm': [],
            'memory_used': [],
            'samples_per_sec': [],
            'time_elapsed': [],
            'worker_id': []  # Track which worker produced each metric
        }
        
        self.start_time = datetime.now()
        
        # Create interactive dashboard
        self.fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Training & Validation Loss',
                'Training & Validation Accuracy',
                'Learning Rate',
                'Gradient Norm',
                'Memory Usage (GB)',
                'Training Speed'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Set figure size and style
        self.fig.update_layout(
            height=1200,
            width=1600,
            showlegend=True,
            template='plotly_dark',
            title={
                'text': 'Training Progress Dashboard',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            }
        )
    
    def update(self, epoch: int, train_loss: float, val_loss: float, 
               train_accuracy: float, val_accuracy: float,
               learning_rate: float, gradient_norm: float, batch_size: int,
               num_samples: int, worker_id: int) -> None:
        """Update metrics with new values"""
        # Calculate time elapsed
        time_elapsed = (datetime.now() - self.start_time).total_seconds() / 3600  # hours
        
        # Calculate memory usage
        memory_used = psutil.Process(os.getpid()).memory_info().rss / 1024**3  # GB
        
        # Calculate training speed
        samples_per_sec = num_samples / (time_elapsed * 3600)  # samples per second
        
        # Update metrics
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_accuracy'].append(train_accuracy)
        self.metrics['val_accuracy'].append(val_accuracy)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['gradient_norm'].append(gradient_norm)
        self.metrics['memory_used'].append(memory_used)
        self.metrics['samples_per_sec'].append(samples_per_sec)
        self.metrics['time_elapsed'].append(time_elapsed)
        self.metrics['worker_id'].append(worker_id)
        
        # Save metrics to JSON
        self._save_metrics()
    
    def _save_metrics(self) -> None:
        """Save metrics to JSON file"""
        metrics_file = os.path.join(self.save_dir, 'training_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def plot(self) -> None:
        """Create and save interactive plots"""
        # Clear previous traces
        self.fig.data = []
        
        # Convert metrics to DataFrame for easier manipulation
        df = pd.DataFrame(self.metrics)
        
        # Find best model's metrics (lowest validation loss)
        best_worker_metrics = {}
        for worker_id in df['worker_id'].unique():
            worker_df = df[df['worker_id'] == worker_id]
            best_epoch_idx = worker_df['val_loss'].idxmin()
            best_worker_metrics[worker_id] = worker_df.loc[best_epoch_idx]
        
        # Get the overall best worker
        best_worker_id = min(best_worker_metrics, key=lambda x: best_worker_metrics[x]['val_loss'])
        best_worker_df = df[df['worker_id'] == best_worker_id]
        
        # Create subplots with shared x-axis
        self.fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Training & Validation Loss',
                'Training & Validation Accuracy',
                'Learning Rate',
                'Gradient Norm',
                'Memory Usage (GB)',
                'Training Speed'
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            shared_xaxes=True
        )
        
        # 1. Training & Validation Loss
        self.fig.add_trace(
            go.Scatter(x=best_worker_df['epoch'], y=best_worker_df['train_loss'],
                      name='Train Loss', line=dict(color='#00ff00', width=2)),
            row=1, col=1
        )
        self.fig.add_trace(
            go.Scatter(x=best_worker_df['epoch'], y=best_worker_df['val_loss'],
                      name='Val Loss', line=dict(color='#ff0000', width=2)),
            row=1, col=1
        )
        
        # 2. Training & Validation Accuracy
        self.fig.add_trace(
            go.Scatter(x=best_worker_df['epoch'], y=best_worker_df['train_accuracy'],
                      name='Train Accuracy', line=dict(color='#00ff00', width=2)),
            row=1, col=2
        )
        self.fig.add_trace(
            go.Scatter(x=best_worker_df['epoch'], y=best_worker_df['val_accuracy'],
                      name='Val Accuracy', line=dict(color='#ff0000', width=2)),
            row=1, col=2
        )
        
        # 3. Learning Rate
        self.fig.add_trace(
            go.Scatter(x=best_worker_df['epoch'], y=best_worker_df['learning_rate'],
                      name='Learning Rate', line=dict(color='#00ffff', width=2)),
            row=2, col=1
        )
        
        # 4. Gradient Norm
        self.fig.add_trace(
            go.Scatter(x=best_worker_df['epoch'], y=best_worker_df['gradient_norm'],
                      name='Gradient Norm', line=dict(color='#ffff00', width=2)),
            row=2, col=2
        )
        
        # 5. Memory Usage
        self.fig.add_trace(
            go.Scatter(x=best_worker_df['epoch'], y=best_worker_df['memory_used'],
                      name='Memory (GB)', line=dict(color='#ff00ff', width=2)),
            row=3, col=1
        )
        
        # 6. Training Speed
        self.fig.add_trace(
            go.Scatter(x=best_worker_df['epoch'], y=best_worker_df['samples_per_sec'],
                      name='Samples/sec', line=dict(color='#ffffff', width=2)),
            row=3, col=2
        )
        
        # Update axes labels and ranges
        self.fig.update_xaxes(title_text="Epoch", row=1, col=1)
        self.fig.update_xaxes(title_text="Epoch", row=1, col=2)
        self.fig.update_xaxes(title_text="Epoch", row=2, col=1)
        self.fig.update_xaxes(title_text="Epoch", row=2, col=2)
        self.fig.update_xaxes(title_text="Epoch", row=3, col=1)
        self.fig.update_xaxes(title_text="Epoch", row=3, col=2)
        
        self.fig.update_yaxes(title_text="Loss", row=1, col=1)
        self.fig.update_yaxes(title_text="Accuracy", row=1, col=2, range=[0, 1])
        self.fig.update_yaxes(title_text="Learning Rate", row=2, col=1)
        self.fig.update_yaxes(title_text="Gradient Norm", row=2, col=2)
        self.fig.update_yaxes(title_text="Memory (GB)", row=3, col=1)
        self.fig.update_yaxes(title_text="Samples/sec", row=3, col=2)
        
        # Update layout
        self.fig.update_layout(
            height=1200,
            width=1600,
            showlegend=True,
            template='plotly_dark',
            title={
                'text': f'Training Progress (Best Worker {best_worker_id})',
                'y':0.98,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            margin=dict(
                l=50,
                r=50,
                t=100,
                b=50
            ),
            font=dict(
                size=14
            )
        )
        
        # Save interactive HTML
        self.fig.write_html(os.path.join(self.save_dir, 'training_progress.html'))
        
        # Save static image with higher resolution
        self.fig.write_image(os.path.join(self.save_dir, 'training_progress.png'), 
                           scale=2, width=1600, height=1200)
    
    def plot_prediction_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                               epoch: int) -> None:
        """Create prediction analysis plots"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Predictions vs Actual',
                'Residuals',
                'Error Distribution',
                'Q-Q Plot'
            )
        )
        
        # 1. Predictions vs Actual
        fig.add_trace(
            go.Scatter(x=y_true.flatten(), y=y_pred.flatten(), mode='markers',
                      name='Predictions', marker=dict(color='#00ff00', size=2)),
            row=1, col=1
        )
        
        # Add diagonal line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      name='Perfect Prediction', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # 2. Residuals
        residuals = y_pred.flatten() - y_true.flatten()
        fig.add_trace(
            go.Scatter(x=y_pred.flatten(), y=residuals, mode='markers',
                      name='Residuals', marker=dict(color='#ff0000', size=2)),
            row=1, col=2
        )
        
        # Add horizontal line at y=0
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[0, 0],
                      name='Zero Line', line=dict(color='white', dash='dash')),
            row=1, col=2
        )
        
        # 3. Error Distribution
        fig.add_trace(
            go.Histogram(x=residuals, name='Error Distribution',
                        nbinsx=50, opacity=0.7),
            row=2, col=1
        )
        
        # 4. Q-Q Plot
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = np.random.normal(0, 1, len(residuals))
        theoretical_quantiles.sort()
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_residuals,
                      mode='markers', name='Q-Q Plot',
                      marker=dict(color='#00ffff', size=2)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1400,
            showlegend=True,
            template='plotly_dark',
            title={
                'text': f'Prediction Analysis (Epoch {epoch})',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            }
        )
        
        # Save plots
        fig.write_html(os.path.join(self.save_dir, f'prediction_analysis_epoch_{epoch}.html'))
        fig.write_image(os.path.join(self.save_dir, f'prediction_analysis_epoch_{epoch}.png'))
    
    def save_summary(self) -> None:
        """Save training summary"""
        summary = {
            'total_epochs': len(self.metrics['epoch']),
            'best_train_loss': min(self.metrics['train_loss']),
            'best_val_loss': min(self.metrics['val_loss']),
            'total_time_hours': self.metrics['time_elapsed'][-1],
            'final_learning_rate': self.metrics['learning_rate'][-1],
            'peak_memory_gb': max(self.metrics['memory_used']),
            'avg_samples_per_sec': np.mean(self.metrics['samples_per_sec'])
        }
        
        # Save summary to file
        summary_file = os.path.join(self.save_dir, 'training_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        return summary 