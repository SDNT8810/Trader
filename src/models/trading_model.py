import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import argparse
from typing import Dict, Any, Optional

# Default configuration
DEFAULT_CONFIG = {
    'model': {
        'name': "TradingANN",
        'input_dim': 100,  # m*n
        'hidden_dims': [256, 128, 64, 32, 16],  # 5 hidden layers
        'output_dim': 1,  # l
        'activation': "relu",  # activation function for hidden layers
        'output_activation': "tanh",  # activation function for output layer
        'weight_init': "xavier"  # weight initialization method
    }
}

class TradingANN(nn.Module):
    """
    Fully Connected Artificial Neural Network for Forex Trading
    
    Architecture:
    - Input: m*n dimensional input
    - 5 Hidden Layers
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
        model_config = self.config['model']
        
        # Extract parameters from config
        self.input_dim = model_config['input_dim']
        self.hidden_dims = model_config['hidden_dims']
        self.output_dim = model_config['output_dim']
        self.activation = model_config['activation']
        self.output_activation = model_config['output_activation']
        self.weight_init = model_config['weight_init']
        
        # Validate input parameters
        if len(self.hidden_dims) != 5:
            raise ValueError("Model requires exactly 5 hidden layers")
        
        # Input layer
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dims[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1])
            for i in range(len(self.hidden_dims)-1)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using specified initialization method"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.weight_init == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif self.weight_init == "kaiming":
                    nn.init.kaiming_uniform_(m.weight)
                else:
                    # Default to xavier if unknown method
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _get_activation(self, name: str):
        """Get activation function by name"""
        activations = {
            "relu": F.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "leaky_relu": F.leaky_relu
        }
        return activations.get(name.lower(), F.relu)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim) with values in [-1, 1]
        """
        activation_fn = self._get_activation(self.activation)
        output_activation_fn = self._get_activation(self.output_activation)
        
        # Input layer
        x = activation_fn(self.input_layer(x))
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = activation_fn(layer(x))
        
        # Output layer
        x = output_activation_fn(self.output_layer(x))
        
        return x

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_model(config_path: Optional[str] = None) -> TradingANN:
    """
    Create model from configuration file or default configuration
    
    Parameters:
    -----------
    config_path : str, optional
        Path to configuration file. If not provided, uses default configuration.
    """
    if config_path is not None:
        config = load_config(config_path)
    else:
        config = DEFAULT_CONFIG
    return TradingANN(config)

def print_model_info(model: TradingANN, config_source: str = "default configuration"):
    """Print model information"""
    print(f"Model created using {config_source}")
    print("\nModel Configuration:")
    print(f"Input dimension: {model.input_dim}")
    print(f"Hidden dimensions: {model.hidden_dims}")
    print(f"Output dimension: {model.output_dim}")
    print(f"Activation (hidden): {model.activation}")
    print(f"Activation (output): {model.output_activation}")
    print(f"Weight initialization: {model.weight_init}")
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

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