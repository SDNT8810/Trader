import torch
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Arrow, FancyArrowPatch
import matplotlib.patches as patches

# Add the parent directory to the path to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from main import TimeSeriesModel

def create_model_architecture_plot(save_path):
    """Create a detailed visualization of the model architecture."""
    fig, ax = plt.subplots(figsize=(25, 15))
    
    # Define layer properties
    layer_properties = {
        'input': {'x': 0.1, 'y': 0.5, 'width': 0.1, 'height': 0.3, 'color': '#3498db'},
        'lstm1': {'x': 0.25, 'y': 0.3, 'width': 0.1, 'height': 0.3, 'color': '#2ecc71'},
        'lstm2': {'x': 0.25, 'y': 0.5, 'width': 0.1, 'height': 0.3, 'color': '#2ecc71'},
        'lstm3': {'x': 0.25, 'y': 0.7, 'width': 0.1, 'height': 0.3, 'color': '#2ecc71'},
        'lstm4': {'x': 0.35, 'y': 0.3, 'width': 0.1, 'height': 0.3, 'color': '#2ecc71'},
        'lstm5': {'x': 0.35, 'y': 0.5, 'width': 0.1, 'height': 0.3, 'color': '#2ecc71'},
        'lstm6': {'x': 0.35, 'y': 0.7, 'width': 0.1, 'height': 0.3, 'color': '#2ecc71'},
        'lstm7': {'x': 0.45, 'y': 0.3, 'width': 0.1, 'height': 0.3, 'color': '#2ecc71'},
        'lstm8': {'x': 0.45, 'y': 0.5, 'width': 0.1, 'height': 0.3, 'color': '#2ecc71'},
        'lstm9': {'x': 0.45, 'y': 0.7, 'width': 0.1, 'height': 0.3, 'color': '#2ecc71'},
        'lstm10': {'x': 0.55, 'y': 0.5, 'width': 0.1, 'height': 0.3, 'color': '#2ecc71'},
        'attention': {'x': 0.65, 'y': 0.5, 'width': 0.1, 'height': 0.3, 'color': '#e74c3c'},
        'output1': {'x': 0.75, 'y': 0.5, 'width': 0.1, 'height': 0.3, 'color': '#f1c40f'},
        'output2': {'x': 0.85, 'y': 0.5, 'width': 0.1, 'height': 0.3, 'color': '#f1c40f'}
    }
    
    # Draw layers
    for layer, props in layer_properties.items():
        rect = patches.Rectangle(
            (props['x'], props['y'] - props['height']/2),
            props['width'],
            props['height'],
            linewidth=2,
            edgecolor='black',
            facecolor=props['color'],
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add layer text
        if layer == 'input':
            text = f'Input Layer\n(50, 101)'
        elif layer.startswith('lstm'):
            text = f'LSTM Layer\n256 units\nBidirectional'
        elif layer == 'attention':
            text = 'Attention\nMechanism'
        elif layer == 'output1':
            text = 'Output Layer 1\n256 → 128'
        else:
            text = 'Output Layer 2\n128 → 1'
            
        ax.text(props['x'] + props['width']/2,
                props['y'],
                text,
                ha='center',
                va='center',
                fontsize=10,
                fontweight='bold')
    
    # Draw connections
    connections = [
        ('input', 'lstm1'), ('input', 'lstm2'), ('input', 'lstm3'),
        ('lstm1', 'lstm4'), ('lstm2', 'lstm5'), ('lstm3', 'lstm6'),
        ('lstm4', 'lstm7'), ('lstm5', 'lstm8'), ('lstm6', 'lstm9'),
        ('lstm7', 'lstm10'), ('lstm8', 'lstm10'), ('lstm9', 'lstm10'),
        ('lstm10', 'attention'),
        ('attention', 'output1'),
        ('output1', 'output2')
    ]
    
    for start, end in connections:
        start_props = layer_properties[start]
        end_props = layer_properties[end]
        
        # Calculate arrow start and end points
        start_x = start_props['x'] + start_props['width']
        start_y = start_props['y']
        end_x = end_props['x']
        end_y = end_props['y']
        
        # Draw arrow
        arrow = FancyArrowPatch(
            (start_x, start_y),
            (end_x, end_y),
            arrowstyle='->',
            linewidth=1.5,
            color='black',
            connectionstyle='arc3,rad=0.2'
        )
        ax.add_patch(arrow)
    
    # Add layer details
    details = [
        (0.1, 0.9, 'Input Features:\n- 50 time steps\n- 101 features per step'),
        (0.25, 0.9, 'LSTM Processing:\n- 10 stacked layers\n- 256 hidden units\n- Bidirectional\n- Dropout: 0.2'),
        (0.55, 0.9, 'Attention:\n- Weighted importance\n- Context vector generation'),
        (0.8, 0.9, 'Output Processing:\n- Deep neural network\n- Multiple dense layers\n- Dropout regularization')
    ]
    
    for x, y, text in details:
        ax.text(x, y, text,
                ha='center',
                va='center',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Set plot properties
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title('Deep Time Series Model Architecture for Gold Price Prediction', fontsize=16, pad=20)
    
    # Save the figure
    plt.savefig(os.path.join(save_path, 'model_architecture.png'), 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

def create_data_flow_plot(save_path):
    """Create a visualization of the moving window data flow."""
    plt.figure(figsize=(15, 8))
    
    # Create sample time series data
    time_points = 200
    features = 5  # Show only 5 features for clarity
    data = np.random.randn(time_points, features)
    
    # Plot the time series
    for i in range(features):
        plt.plot(data[:, i], label=f'Feature {i+1}', alpha=0.7)
    
    # Highlight a moving window
    window_size = 50
    window_start = 75
    window_end = window_start + window_size
    
    # Add window rectangle
    plt.axvspan(window_start, window_end, color='yellow', alpha=0.2)
    plt.text((window_start + window_end)/2, plt.ylim()[1]*0.95,
             f'Window Size: {window_size}',
             ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Add arrows showing window movement
    plt.arrow(window_end, plt.ylim()[0]*0.8,
              window_size/2, 0,
              head_width=0.5, head_length=2, fc='k', ec='k')
    
    plt.title('Moving Window Data Flow')
    plt.xlabel('Time Steps')
    plt.ylabel('Feature Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'data_flow.png'))
    plt.close()

def create_attention_weights_plot(save_path):
    """Create a visualization of attention weights."""
    plt.figure(figsize=(12, 6))
    
    # Create sample attention weights
    window_size = 50
    attention_weights = np.random.rand(window_size)
    attention_weights = attention_weights / attention_weights.sum()
    
    # Plot attention weights
    plt.bar(range(window_size), attention_weights, alpha=0.7)
    plt.plot(range(window_size), attention_weights, 'r-', linewidth=2)
    
    plt.title('Attention Weights Across Time Steps')
    plt.xlabel('Time Step in Window')
    plt.ylabel('Attention Weight')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'attention_weights.png'))
    plt.close()

def visualize_model(save_path=None):
    """
    Create comprehensive visualizations of the Time Series Model.
    
    Args:
        save_path (str): Directory where the visualizations will be saved
    """
    if save_path is None:
        save_path = os.path.join(parent_dir, 'Figs/Model')
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Create visualizations
    create_model_architecture_plot(save_path)
    create_data_flow_plot(save_path)
    create_attention_weights_plot(save_path)
    
    # Print model information
    print("\nModel Architecture Information:")
    print(f"Window size: 50")
    print(f"Number of features: 101")
    print(f"LSTM layers: 10")
    print(f"Hidden size: 256")
    print(f"Dropout rate: 0.2")
    
    print(f"\nVisualizations saved in {save_path}:")
    print(f"1. Model architecture: model_architecture.png")
    print(f"2. Data flow: data_flow.png")
    print(f"3. Attention weights: attention_weights.png")

if __name__ == "__main__":
    visualize_model() 