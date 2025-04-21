import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import argparse
from typing import Dict, Any, Optional
import os

# Default configuration with ALL possible parameters
DEFAULT_CONFIG = {
    'model': {
        'name': "TradingANN",
        'input_dim': 1720,  # window_size * num_features = 60 * 43
        'hidden_dims': [512, 512, 256, 128, 64],  # 5 hidden layers
        'output_dim': 11,  # 10 changes + total change
        'activation': "leaky_relu",  # activation function for hidden layers
        'output_activation': "tanh",  # activation function for output layer
        'weight_init': "kaiming",  # weight initialization method
        'dropout': 0.3,  # dropout rate
        'batch_norm': True,  # whether to use batch normalization
        'layer_norm': False,  # whether to use layer normalization
        'residual': False,  # whether to use residual connections
        'bias': True  # whether to use bias in linear layers
    }
}

class TradingANN(nn.Module):
    """
    Fully Connected Artificial Neural Network for Forex Trading
    
    Architecture:
    - Input: m*n dimensional input (window_size * num_features)
    - 5 Hidden Layers with configurable dimensions
    - Output: l*1 vector with values in range [-1, +1]
    
    Parameters:
    -----------
    config : Dict[str, Any], optional
        Configuration dictionary containing model parameters.
        If not provided, uses default configuration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super(TradingANN, self).__init__()
        
        # Use default config if none provided
        self.config = config if config is not None else DEFAULT_CONFIG
        model_config = self.config.get('model', DEFAULT_CONFIG['model'])
        
        # Extract parameters from config with defaults
        self.input_dim = model_config.get('input_dim', DEFAULT_CONFIG['model']['input_dim'])
        self.hidden_dims = model_config.get('hidden_dims', DEFAULT_CONFIG['model']['hidden_dims'])
        self.output_dim = model_config.get('output_dim', DEFAULT_CONFIG['model']['output_dim'])
        self.activation = model_config.get('activation', DEFAULT_CONFIG['model']['activation'])
        self.output_activation = model_config.get('output_activation', DEFAULT_CONFIG['model']['output_activation'])
        self.weight_init = model_config.get('weight_init', DEFAULT_CONFIG['model']['weight_init'])
        self.dropout = model_config.get('dropout', DEFAULT_CONFIG['model']['dropout'])
        self.use_batch_norm = model_config.get('batch_norm', DEFAULT_CONFIG['model']['batch_norm'])
        self.use_layer_norm = model_config.get('layer_norm', DEFAULT_CONFIG['model']['layer_norm'])
        self.use_residual = model_config.get('residual', DEFAULT_CONFIG['model']['residual'])
        self.use_bias = model_config.get('bias', DEFAULT_CONFIG['model']['bias'])
        
        # LSTM layer for temporal processing
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dims[0],
            num_layers=2,
            batch_first=True,
            dropout=self.dropout if self.dropout > 0 else 0,
            bidirectional=True
        )
        
        # Calculate LSTM output size (bidirectional)
        lstm_output_size = self.hidden_dims[0] * 2
        
        # Input layer (after LSTM)
        self.input_layer = nn.Linear(lstm_output_size, self.hidden_dims[0], bias=self.use_bias)
        if self.use_batch_norm:
            self.bn_input = nn.BatchNorm1d(self.hidden_dims[0])
        if self.use_layer_norm:
            self.ln_input = nn.LayerNorm(self.hidden_dims[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(self.hidden_dims)-1):
            layer_modules = []
            # Linear transformation
            layer_modules.append(
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1], bias=self.use_bias)
            )
            # Normalization
            if self.use_batch_norm:
                layer_modules.append(nn.BatchNorm1d(self.hidden_dims[i+1]))
            if self.use_layer_norm:
                layer_modules.append(nn.LayerNorm(self.hidden_dims[i+1]))
            # Dropout
            if self.dropout > 0:
                layer_modules.append(nn.Dropout(self.dropout))
            # Add all modules for this layer
            self.hidden_layers.extend(layer_modules)
        
        # Output layer
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim, bias=self.use_bias)
        
        # Initialize weights
        self._initialize_weights()
        
        # Print model architecture
        self._print_architecture()
    
    def _initialize_weights(self):
        """Initialize weights using specified initialization method"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.weight_init == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', 
                                         nonlinearity='leaky_relu' if self.activation == 'leaky_relu' else 'relu')
                elif self.weight_init == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif self.weight_init == "uniform":
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _get_activation(self, name: str):
        """Get activation function by name"""
        activations = {
            "relu": F.relu,
            "leaky_relu": lambda x: F.leaky_relu(x, negative_slope=0.01),
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "elu": F.elu,
            "selu": F.selu,
            "gelu": F.gelu
        }
        return activations.get(name.lower(), F.leaky_relu)
    
    def _print_architecture(self):
        """Print model architecture"""
        print("\nModel Architecture:")
        print(f"Input Layer: {self.input_dim} -> {self.hidden_dims[0]}")
        print(f"Normalization: BatchNorm={self.use_batch_norm}, LayerNorm={self.use_layer_norm}")
        for i in range(len(self.hidden_dims)-1):
            print(f"Hidden Layer {i+1}: {self.hidden_dims[i]} -> {self.hidden_dims[i+1]}")
        print(f"Output Layer: {self.hidden_dims[-1]} -> {self.output_dim}")
        print(f"Activation: {self.activation}")
        print(f"Output Activation: {self.output_activation}")
        print(f"Weight Initialization: {self.weight_init}")
        print(f"Dropout Rate: {self.dropout}")
        print(f"Using Bias: {self.use_bias}")
        print(f"Using Residual Connections: {self.use_residual}\n")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, window_size, input_dim)
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim)
        """
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        # Take the last time step's output
        x = lstm_out[:, -1, :]
        
        # Input layer
        x = self.input_layer(x)
        if self.use_batch_norm:
            x = self.bn_input(x)
        if self.use_layer_norm:
            x = self.ln_input(x)
        x = self._get_activation(self.activation)(x)
        
        # Hidden layers
        layer_idx = 0
        modules_per_layer = 1 + int(self.use_batch_norm) + int(self.use_layer_norm) + int(self.dropout > 0)
        
        while layer_idx < len(self.hidden_layers):
            # Save for residual connection
            if self.use_residual:
                identity = x
            
            # Apply layer modules
            for i in range(modules_per_layer):
                if layer_idx + i < len(self.hidden_layers):
                    x = self.hidden_layers[layer_idx + i](x)
            
            # Add residual connection if shapes match
            if self.use_residual and x.shape == identity.shape:
                x = x + identity
            
            layer_idx += modules_per_layer
        
        # Output layer with specified activation
        x = self.output_layer(x)
        if self.output_activation:
            x = self._get_activation(self.output_activation)(x)
        
        return x

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with fallback to defaults
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration file
        
    Returns:
    --------
    Dict[str, Any]
        Configuration dictionary with all parameters set
    """
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}. Using default configuration.")
        return DEFAULT_CONFIG
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Merge with defaults to ensure all parameters exist
        merged_config = DEFAULT_CONFIG.copy()
        if 'model' in config:
            merged_config['model'].update(config['model'])
        
        return merged_config
    except Exception as e:
        print(f"Error loading config file: {str(e)}")
        print("Using default configuration.")
        return DEFAULT_CONFIG

def create_model(config_path: Optional[str] = None) -> TradingANN:
    """
    Create a new TradingANN model with proper parameter handling
    
    Parameters:
    -----------
    config_path : str, optional
        Path to the configuration file. If None, uses default configuration.
        
    Returns:
    --------
    TradingANN
        New TradingANN model instance
    """
    if config_path is None:
        print("No config path provided. Using default configuration.")
        return TradingANN()
    
    config = load_config(config_path)
    return TradingANN(config)

def print_model_info(model: TradingANN, config_source: str = "default configuration"):
    """
    Print model information
    
    Parameters:
    -----------
    model : TradingANN
        Model to print information about
    config_source : str, optional
        Source of the configuration, by default "default configuration"
    """
    print(f"\nModel Information (using {config_source}):")
    print(f"Input Dimension: {model.input_dim}")
    print(f"Hidden Dimensions: {model.hidden_dims}")
    print(f"Output Dimension: {model.output_dim}")
    print(f"Activation: {model.activation}")
    print(f"Output Activation: {model.output_activation}")
    print(f"Weight Initialization: {model.weight_init}")
    print(f"Dropout Rate: {model.dropout}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trading ANN Model')
    parser.add_argument('--config', type=str, required=False,
                      help='Path to configuration file (optional)')
    args = parser.parse_args()
    
    # Create model
    model = create_model(args.config)
    
    # Print model information
    config_source = args.config if args.config else "default configuration"
    print_model_info(model, config_source) 