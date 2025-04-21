import numpy as np
import yaml
from typing import Dict, List, Tuple, Union
import os
import shutil

class DataNormalizer:
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the DataNormalizer with configuration from YAML file.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.normalization_params = {}
        self._validate_config()
    
    def create_fig_directory():
        """Create directory for saving figures if it doesn't exist."""
        fig_dir = 'src/Figs/Normalized'
        if os.path.exists(fig_dir):
            shutil.rmtree(fig_dir)
        os.makedirs(fig_dir)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _validate_config(self) -> None:
        """Validate the configuration structure."""
        if 'normalization' not in self.config:
            raise ValueError("Normalization configuration not found in config file")
        
        if 'methods' not in self.config['normalization']:
            raise ValueError("Normalization methods not specified in config")
        
        if 'indicators' not in self.config['normalization']:
            raise ValueError("Indicator configuration not found in config")
    
    def fit(self, data: Dict[str, np.ndarray]) -> None:
        """
        Calculate normalization parameters for each indicator.
        
        Args:
            data (Dict[str, np.ndarray]): Dictionary containing indicator data
        """
        for indicator_type, columns in self.config['normalization']['indicators'].items():
            for column in columns:
                if column in data:
                    # Calculate min and max for min-max normalization
                    min_val = np.min(data[column])
                    max_val = np.max(data[column])
                    
                    # Calculate mean and std for z-score normalization
                    mean_val = np.mean(data[column])
                    std_val = np.std(data[column])
                    
                    self.normalization_params[column] = {
                        'min': min_val,
                        'max': max_val,
                        'mean': mean_val,
                        'std': std_val
                    }
    
    def transform(self, data: Dict[str, np.ndarray], method: str = 'min_max') -> Dict[str, np.ndarray]:
        """
        Normalize the data using the specified method.
        
        Args:
            data (Dict[str, np.ndarray]): Dictionary containing indicator data
            method (str): Normalization method ('min_max' or 'z_score')
            
        Returns:
            Dict[str, np.ndarray]: Normalized data
        """
        if method not in ['min_max', 'z_score']:
            raise ValueError("Normalization method must be either 'min_max' or 'z_score'")
        
        normalized_data = {}
        for indicator_type, columns in self.config['normalization']['indicators'].items():
            for column in columns:
                if column in data:
                    if method == 'min_max':
                        normalized_data[column] = self._min_max_normalize(
                            data[column],
                            self.normalization_params[column]['min'],
                            self.normalization_params[column]['max']
                        )
                    else:  # z_score
                        normalized_data[column] = self._z_score_normalize(
                            data[column],
                            self.normalization_params[column]['mean'],
                            self.normalization_params[column]['std']
                        )
        
        return normalized_data
    
    def inverse_transform(self, data: Dict[str, np.ndarray], method: str = 'min_max') -> Dict[str, np.ndarray]:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data (Dict[str, np.ndarray]): Dictionary containing normalized data
            method (str): Normalization method used ('min_max' or 'z_score')
            
        Returns:
            Dict[str, np.ndarray]: Original scale data
        """
        if method not in ['min_max', 'z_score']:
            raise ValueError("Normalization method must be either 'min_max' or 'z_score'")
        
        original_data = {}
        for column in data:
            if method == 'min_max':
                original_data[column] = self._min_max_denormalize(
                    data[column],
                    self.normalization_params[column]['min'],
                    self.normalization_params[column]['max']
                )
            else:  # z_score
                original_data[column] = self._z_score_denormalize(
                    data[column],
                    self.normalization_params[column]['mean'],
                    self.normalization_params[column]['std']
                )
        
        return original_data
    
    def _min_max_normalize(self, data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Apply min-max normalization."""
        range_min, range_max = self.config['normalization']['methods']['min_max']['range']
        return range_min + (data - min_val) * (range_max - range_min) / (max_val - min_val)
    
    def _z_score_normalize(self, data: np.ndarray, mean_val: float, std_val: float) -> np.ndarray:
        """Apply z-score normalization."""
        epsilon = self.config['normalization']['methods']['z_score']['epsilon']
        return (data - mean_val) / (std_val + epsilon)
    
    def _min_max_denormalize(self, data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Inverse min-max normalization."""
        range_min, range_max = self.config['normalization']['methods']['min_max']['range']
        return min_val + (data - range_min) * (max_val - min_val) / (range_max - range_min)
    
    def _z_score_denormalize(self, data: np.ndarray, mean_val: float, std_val: float) -> np.ndarray:
        """Inverse z-score normalization."""
        epsilon = self.config['normalization']['methods']['z_score']['epsilon']
        return data * (std_val + epsilon) + mean_val
    
    def save_params(self, path: str = "data/normalization_params.npz") -> None:
        """Save normalization parameters to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **self.normalization_params)
    
    def load_params(self, path: str = "data/normalization_params.npz") -> None:
        """Load normalization parameters from file."""
        if os.path.exists(path):
            params = np.load(path)
            self.normalization_params = {k: params[k].item() for k in params.files}
        else:
            raise FileNotFoundError(f"Normalization parameters file not found at {path}") 
        


if __name__ == "__main__":
    # Set style for better visualization
    plt.style.use('ggplot')
    plt.rcParams['figure.figsize'] = (15, 8)
    plt.rcParams['font.size'] = 10
    
    # Plot all normalized data
    plot_normalized_data()