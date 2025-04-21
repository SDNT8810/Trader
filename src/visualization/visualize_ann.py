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
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Generate visualization
    dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
    
    if save_path:
        dot.render(save_path, format='png', cleanup=True)
        print(f"Model architecture visualization saved to {save_path}.png")
    else:
        dot.view()

def plot_layer_sizes(model: nn.Module, save_path: Optional[str] = None) -> None:
    """
    Create a bar plot showing the size of each layer
    
    Parameters:
    -----------
    model : nn.Module
        The neural network model
    save_path : Optional[str]
        Path to save the plot. If None, shows the plot.
    """
    # Get layer sizes
    layer_sizes = []
    layer_names = []
    
    # Input layer
    layer_sizes.append(model.input_dim)
    layer_names.append('Input')
    
    # Hidden layers
    for i, layer in enumerate(model.hidden_layers):
        layer_sizes.append(layer.out_features)
        layer_names.append(f'Hidden {i+1}')
    
    # Output layer
    layer_sizes.append(model.output_dim)
    layer_names.append('Output')
    
    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(layer_names, layer_sizes)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.title('Neural Network Layer Sizes')
    plt.xlabel('Layer')
    plt.ylabel('Number of Neurons')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Layer sizes plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_weights(model: nn.Module, save_path: Optional[str] = None) -> None:
    """
    Visualize the weight distributions of each layer
    
    Parameters:
    -----------
    model : nn.Module
        The neural network model
    save_path : Optional[str]
        Path to save the plot. If None, shows the plot.
    """
    # Get all weights
    weights = []
    layer_names = []
    
    # Input layer
    weights.append(model.input_layer.weight.detach().numpy().flatten())
    layer_names.append('Input Layer')
    
    # Hidden layers
    for i, layer in enumerate(model.hidden_layers):
        weights.append(layer.weight.detach().numpy().flatten())
        layer_names.append(f'Hidden Layer {i+1}')
    
    # Output layer
    weights.append(model.output_layer.weight.detach().numpy().flatten())
    layer_names.append('Output Layer')
    
    # Create plot
    plt.figure(figsize=(15, 5 * len(weights)))
    for i, (w, name) in enumerate(zip(weights, layer_names)):
        plt.subplot(len(weights), 1, i+1)
        plt.hist(w, bins=50, alpha=0.7)
        plt.title(f'{name} Weight Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Weight distributions plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize ANN Model')
    parser.add_argument('--config', type=str, required=False,
                      help='Path to configuration file (optional)')
    parser.add_argument('--output_dir', type=str, default='Figs',
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