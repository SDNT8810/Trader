import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class TradingModel:
    def __init__(self, input_shape, learning_rate=0.001):
        """
        Initialize the trading model.
        
        Args:
            input_shape (tuple): Shape of input data (sequence_length, n_features)
            learning_rate (float): Learning rate for the optimizer
        """
        self.model = self._build_model(input_shape, learning_rate)
        self.history = None
        
    def _build_model(self, input_shape, learning_rate):
        """Build the GRU model architecture."""
        model = Sequential([
            Input(shape=input_shape),
            BatchNormalization(),
            GRU(256, return_sequences=True, activation='tanh'),
            Dropout(0.3),
            BatchNormalization(),
            GRU(128, return_sequences=True, activation='tanh'),
            Dropout(0.3),
            BatchNormalization(),
            GRU(64, return_sequences=False, activation='tanh'),
            Dropout(0.3),
            BatchNormalization(),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')  # Output probabilities for sell, hold, buy
        ])
        
        def custom_loss(y_true, y_pred):
            # Convert one-hot encoded true values
            y_true_cat = tf.cast(y_true + 1, tf.int32)  # Convert from [-1,0,1] to [0,1,2]
            y_true_1hot = tf.one_hot(y_true_cat, depth=3)
            
            # Categorical crossentropy
            cce = tf.keras.losses.categorical_crossentropy(y_true_1hot, y_pred)
            
            # Market-aware class weighting
            # Higher weight for buy signals (reflecting market trend)
            # Very high weight for sell signals (only when very confident)
            class_weights = tf.constant([[4.0, 1.0, 2.0]])  # [sell, hold, buy]
            sample_weights = tf.reduce_sum(y_true_1hot * class_weights, axis=1)
            
            # Add confidence penalty for sell signals
            sell_confidence_penalty = tf.reduce_mean(tf.maximum(0.0, 0.7 - y_pred[:, 0]))
            
            return tf.reduce_mean(cce * sample_weights) + 0.2 * sell_confidence_penalty
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=custom_loss,
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                min_delta=1e-4
            ),
            ModelCheckpoint(
                'best_model.h5',
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # Market-aware class weights
        class_counts = np.bincount(y_train + 1)  # Convert from [-1,0,1] to [0,1,2]
        total_samples = len(y_train)
        
        # Adjust weights to reflect market bias
        # Higher weight for sell to ensure high confidence
        # Moderate weight for buy to maintain natural distribution
        class_weights = {
            0: total_samples / (2 * class_counts[0]),  # Sell - higher weight
            1: total_samples / (3 * class_counts[1]),  # Hold - standard weight
            2: total_samples / (3 * class_counts[2])   # Buy - standard weight
        }
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions on new data."""
        probabilities = self.model.predict(X, verbose=1)
        # Convert probabilities to -1, 0, 1 predictions
        return np.argmax(probabilities, axis=1) - 1  # Convert [0,1,2] to [-1,0,1]
    
    def plot_training_history(self, save_path=None):
        """Plot training and validation loss."""
        if self.history is None:
            logger.warning("No training history available")
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def evaluate_signals(self, y_true, y_pred, threshold=None):
        """
        Evaluate trading signals with profit-focused metrics.
        
        Args:
            y_true: True returns
            y_pred: Predicted signals
            threshold: Not used in this version as predictions are already discrete
            
        Returns:
            dict: Evaluation metrics
        """
        # Calculate basic metrics
        accuracy = np.mean(y_true == y_pred)
        
        # Calculate confusion matrix metrics
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        true_negatives = np.sum((y_true == -1) & (y_pred == -1))
        false_positives = np.sum((y_true != 1) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred != 1))
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate profit-focused metrics
        # Assuming y_true contains actual returns
        strategy_returns = np.where(y_pred == 1, y_true, 
                                  np.where(y_pred == -1, -y_true, 0))
        
        total_return = np.sum(strategy_returns)
        win_rate = np.mean(strategy_returns > 0)
        avg_win = np.mean(strategy_returns[strategy_returns > 0]) if np.any(strategy_returns > 0) else 0
        avg_loss = np.mean(strategy_returns[strategy_returns < 0]) if np.any(strategy_returns < 0) else 0
        profit_factor = abs(np.sum(strategy_returns[strategy_returns > 0]) / 
                          np.sum(strategy_returns[strategy_returns < 0])) if np.any(strategy_returns < 0) else float('inf')
        
        # Calculate Sharpe Ratio (assuming daily returns)
        excess_returns = strategy_returns - 0.02/252  # Assuming 2% risk-free rate annualized
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if len(excess_returns) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_signals': np.sum(y_pred != 0),
            'buy_signals': np.sum(y_pred == 1),
            'sell_signals': np.sum(y_pred == -1),
            'hold_signals': np.sum(y_pred == 0),
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _calculate_profit_factor(self, returns, decisions):
        """Calculate the profit factor of the strategy."""
        strategy_returns = returns * decisions
        gross_profit = np.sum(strategy_returns[strategy_returns > 0])
        gross_loss = abs(np.sum(strategy_returns[strategy_returns < 0]))
        
        return gross_profit / gross_loss if gross_loss != 0 else float('inf') 