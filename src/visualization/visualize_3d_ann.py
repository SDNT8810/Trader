from manim import *
import numpy as np
import torch
import sys
import os
from pathlib import Path
import yaml
import argparse

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from models.trading_model import TradingANN, create_model

class NeuralNetworkScene(ThreeDScene):
    """A 3D scene for visualizing the neural network architecture"""
    
    def __init__(self, model: TradingANN, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.layer_spacing = 4
        self.neuron_spacing = 0.5
        self.neuron_radius = 0.1
        self.layer_colors = [BLUE, GREEN, RED]
    
    def create_layer_block(self, size: int, layer_index: int, label: str) -> VGroup:
        """Create a 3D block representing a layer"""
        height = size * self.neuron_spacing
        width = 2
        depth = 1
        
        # Create the block
        block = Cube(side_length=1)
        block.set_height(height)
        block.set_width(width)
        block.set_depth(depth)
        
        # Position the block
        block.move_to(RIGHT * (layer_index * self.layer_spacing))
        
        # Color the block
        color = self.layer_colors[min(layer_index, len(self.layer_colors)-1)]
        block.set_color(color)
        block.set_opacity(0.3)
        
        # Add label
        label = Text(f"{label}\n({size})", font_size=24)
        label.next_to(block, DOWN)
        
        return VGroup(block, label)
    
    def create_weight_matrix(self, weights: torch.Tensor, start_pos: np.ndarray, end_pos: np.ndarray) -> VGroup:
        """Create a visualization of the weight matrix between layers"""
        matrix = weights.detach().numpy()
        height, width = matrix.shape
        
        # Create heatmap
        heatmap = Rectangle(height=height*0.1, width=width*0.1)
        heatmap.set_fill(WHITE, opacity=0.5)
        heatmap.move_to((start_pos + end_pos) / 2)
        
        # Add color based on weights
        colors = []
        for i in range(height):
            for j in range(width):
                value = matrix[i, j]
                # Normalize value to [0,1] range
                normalized_value = (value + 1) / 2  # Assuming weights are in [-1,1]
                # Create RGB color based on value
                color = rgb_to_color([normalized_value, 1-normalized_value, 0])
                colors.append(color)
        
        # Set colors in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(colors), batch_size):
            batch = colors[i:i+batch_size]
            heatmap.set_color_by_gradient(*batch)
        
        return heatmap
    
    def construct(self):
        """Construct the neural network visualization"""
        # Set up the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.2)
        
        # Create layers
        layers = []
        layer_sizes = [self.model.input_dim] + \
                     [layer.out_features for layer in self.model.hidden_layers] + \
                     [self.model.output_dim]
        layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(self.model.hidden_layers))] + ['Output']
        
        for i, (size, name) in enumerate(zip(layer_sizes, layer_names)):
            layer = self.create_layer_block(size, i, name)
            layers.append(layer)
        
        # Create weight matrices
        weights = []
        weights.append(self.create_weight_matrix(
            self.model.input_layer.weight,
            layers[0][0].get_center(),
            layers[1][0].get_center()
        ))
        
        for i, layer in enumerate(self.model.hidden_layers):
            weights.append(self.create_weight_matrix(
                layer.weight,
                layers[i+1][0].get_center(),
                layers[i+2][0].get_center()
            ))
        
        # Animation sequence
        # 1. Show layers one by one
        self.play(Create(layers[0]))
        for i in range(1, len(layers)):
            self.play(
                Create(layers[i]),
                Create(weights[i-1])
            )
        
        # 2. Rotate and zoom to show 3D structure
        self.wait(2)
        self.move_camera(phi=45 * DEGREES, theta=45 * DEGREES)
        self.wait(2)
        
        # 3. Highlight forward pass
        highlight = Square(side_length=0.3, color=YELLOW)
        highlight.set_opacity(0.5)
        
        for i in range(len(layers)-1):
            self.play(
                highlight.animate.move_to(layers[i][0].get_center()),
                run_time=1
            )
            self.play(
                highlight.animate.move_to(weights[i].get_center()),
                run_time=1
            )
            self.play(
                highlight.animate.move_to(layers[i+1][0].get_center()),
                run_time=1
            )
        
        # 4. Final rotation
        self.wait(2)
        self.stop_ambient_camera_rotation()
        self.move_camera(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.wait(2)

def main():
    parser = argparse.ArgumentParser(description='Create 3D ANN Visualization')
    parser.add_argument('--config', type=str, required=False,
                      help='Path to configuration file (optional)')
    parser.add_argument('--quality', type=str, default='medium_quality',
                      choices=['low_quality', 'medium_quality', 'high_quality'],
                      help='Quality of the rendered animation')
    args = parser.parse_args()
    
    # Create model
    model = create_model(args.config)
    
    # Create and render scene
    scene = NeuralNetworkScene(model)
    scene.render()

if __name__ == "__main__":
    main() 