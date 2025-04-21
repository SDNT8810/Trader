import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchviz import make_dot
from typing import Optional, Tuple
import argparse
import yaml
import os
import sys
from pathlib import Path

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from models.trading_model import TradingANN, create_model

def visualize_model_architecture(
    model: nn.Module,
    input_shape: Tuple[int, int],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the model architecture using torchviz
    
    Parameters:
    -----------
    model : nn.Module
        The neural network model to visualize
    input_shape : Tuple[int, int]
        Shape of the input tensor (batch_size, input_dim)
    save_path : Optional[str]
        Path to save the visualization. If None, shows the plot.
    """
    try:
        # Create a dummy input with the correct shape
        batch_size = 1
        dummy_input = torch.randn(batch_size, model.input_dim)
        
        # Create the visualization
        dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
        dot.render(save_path, format='png', cleanup=True)
        print(f"Model architecture visualization saved to {save_path}.png")
    except Exception as e:
        print(f"Error creating model visualization: {e}")

def plot_layer_sizes(model: nn.Module, save_path: Optional[str] = None) -> None:
    """Plot the sizes of each layer in the model."""
    layer_sizes = []
    layer_names = []
    
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            layer_sizes.append(layer.out_features)
            layer_names.append(name)
        elif isinstance(layer, nn.LSTM):
            layer_sizes.append(layer.hidden_size)
            layer_names.append(name)
    
    # Add input size
    layer_sizes.insert(0, model.input_dim)
    layer_names.insert(0, 'input')
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(layer_sizes)), layer_sizes, 'bo-', linewidth=2, markersize=10)
    plt.grid(True)
    plt.title('Model Layer Sizes')
    plt.xlabel('Layer')
    plt.ylabel('Number of Units')
    plt.xticks(range(len(layer_names)), layer_names, rotation=45)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Layer sizes plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_weights(model: nn.Module, save_path: Optional[str] = None) -> None:
    """Visualize the distribution of weights in the model."""
    weights = []
    layer_names = []
    
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.detach().cpu().numpy().flatten())
            layer_names.append(name)
    
    if not weights:
        print("No weights found to visualize")
        return
        
    plt.figure(figsize=(15, 5))
    
    # Plot weight distributions
    plt.subplot(121)
    for w, name in zip(weights, layer_names):
        plt.hist(w, bins=50, alpha=0.5, label=name)
    plt.title('Weight Distributions by Layer')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot weight statistics
    plt.subplot(122)
    stats = []
    for w in weights:
        stats.append([np.mean(w), np.std(w), np.median(w)])
    stats = np.array(stats)
    
    x = np.arange(len(layer_names))
    width = 0.2
    
    plt.bar(x - width, stats[:, 0], width, label='Mean')
    plt.bar(x, stats[:, 1], width, label='Std')
    plt.bar(x + width, stats[:, 2], width, label='Median')
    
    plt.title('Weight Statistics by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Value')
    plt.xticks(x, layer_names, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Weight visualization saved to {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize ANN Model')
    parser.add_argument('--config', type=str, required=False,
                      help='Path to configuration file (optional)')
    parser.add_argument('--output_dir', type=str, default='src/Figs/Model',
                      help='Directory to save visualizations')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    model = create_model(args.config)
    
    # Visualize model architecture
    arch_path = os.path.join(args.output_dir, 'model_architecture')
    visualize_model_architecture(model, (1, model.input_dim), arch_path)
    
    # Plot layer sizes
    layer_path = os.path.join(args.output_dir, 'layer_sizes.png')
    plot_layer_sizes(model, layer_path)
    
    # Visualize weights
    weights_path = os.path.join(args.output_dir, 'weight_distributions.png')
    visualize_weights(model, weights_path)

if __name__ == "__main__":
    main() 