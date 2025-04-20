# Forex Gold Trading Bot Project

## Project Structure
```
src/
├── data/
│   ├── __init__.py
│   ├── data_preprocessor.py    # Data preprocessing and normalization
│   ├── check_data.py          # Data quality verification
│   └── historical_data/        # Store historical data
├── models/
│   ├── __init__.py
│   ├── trading_model.py        # Main trading strategy model
│   ├── model_utils.py         # Model utility functions
│   └── model_evaluation.py    # Model evaluation metrics
├── utils/
│   ├── __init__.py
│   └── technical_indicators.py # Technical analysis tools
├── visualization/
│   ├── __init__.py
│   ├── plot_normalized_data.py # Data visualization tools
│   └── plotter.py             # General plotting utilities
├── Figs/                       # Generated figures and plots
├── main.py                     # Entry point
└── train.py                    # Model training script
```

## Completed Tasks
### Phase 1: Data Management and Preprocessing
- [x] Set up basic project structure
- [x] Implement data preprocessing pipeline
- [x] Add technical indicators
  - [x] Moving Averages (SMA, EMA, WMA, DEMA, TEMA)
  - [x] RSI
  - [x] MACD with multiple parameters
  - [x] Bollinger Bands with multiple deviations
  - [x] CCI
  - [x] MFI
  - [x] WILLR
  - [x] ATR and NATR
  - [x] ADX and DI indicators
  - [x] Volume indicators (OBV, AD, ADOSC, CMF)
  - [x] Stochastic Oscillator with multiple parameters
  - [x] Custom indicators (Price Change, Volatility, Price Range)
  - [x] Heikin Ashi candlesticks
- [x] Implement data normalization
  - [x] Price normalization
  - [x] Technical indicator normalization
  - [x] Proper range handling for different indicator types
- [x] Create visualization tools
  - [x] Group-specific indicator plots
  - [x] All-in-one indicator plot
  - [x] Row-by-row indicator visualization
- [x] Implement data quality checks
  - [x] NaN and infinite value detection
  - [x] Range validation for normalized values
  - [x] Duplicate row detection
  - [x] Data type verification
- [x] Add missing value handling
- [x] Fix normalization issues
  - [x] Volume indicator normalization
  - [x] ADX/ADXR values
  - [x] Stochastic Oscillator ranges
- [x] Optimize preprocessing performance
  - [x] Memory usage optimization
  - [x] Processing speed improvements
  - [x] Deprecated pandas warnings resolution

### Phase 2: Model Development
- [x] Design deep learning architecture
  - [x] Implement 10-layer LSTM network
  - [x] Add bidirectional processing
  - [x] Include attention mechanism
  - [x] Create deep output network
- [x] Set up model visualization
  - [x] Architecture diagram
  - [x] Data flow visualization
  - [x] Attention weights plot

## Current Tasks
### Phase 2: Feature Engineering and Model Development
1. Feature Engineering
   - [ ] Create lagged features
     - [ ] Price lags
     - [ ] Indicator lags
     - [ ] Volume lags
   - [ ] Add rolling statistics
     - [ ] Rolling means and standard deviations
     - [ ] Rolling correlations
     - [ ] Rolling volatility
   - [ ] Cross-indicator features
     - [ ] Indicator interactions
     - [ ] Divergence detection
     - [ ] Pattern recognition
   - [ ] Time-based features
     - [ ] Time of day patterns
     - [ ] Day of week effects
     - [ ] Seasonal patterns

2. Model Training
   - [ ] Implement training pipeline
     - [ ] Data splitting (train/validation/test)
     - [ ] Batch processing
     - [ ] Training loop
   - [ ] Add model evaluation metrics
     - [ ] Accuracy metrics
     - [ ] Risk-adjusted returns
     - [ ] Drawdown analysis
   - [ ] Implement early stopping
   - [ ] Add learning rate scheduling
   - [ ] Create model checkpointing

3. Model Optimization
   - [ ] Hyperparameter tuning
     - [ ] Layer size optimization
     - [ ] Learning rate optimization
     - [ ] Dropout rate optimization
   - [ ] Architecture refinement
     - [ ] Layer connectivity optimization
     - [ ] Attention mechanism tuning
     - [ ] Output network optimization

## Future Phases
### Phase 3: Backtesting and Optimization
- [ ] Create backtesting framework
- [ ] Implement performance metrics
- [ ] Add transaction costs
- [ ] Include slippage simulation
- [ ] Strategy optimization

### Phase 4: Deployment and Monitoring
- [ ] Set up deployment pipeline
- [ ] Implement monitoring system
- [ ] Create alert system
- [ ] Performance tracking

## Current Model Architecture
- Input Layer: (50, 101) - 50 time steps with 101 features
- LSTM Layers: 10 layers with 256 units each
  - Bidirectional processing
  - Dropout rate: 0.2
- Attention Mechanism: Weighted importance across time steps
- Output Network: Deep architecture (256 → 128 → 64 → 1)
- Total Parameters: [To be calculated after training]

## Technical Requirements
- Python 3.8+
- Required Libraries:
  - pandas
  - numpy
  - ta-lib
  - matplotlib
  - seaborn
  - scikit-learn
  - PyTorch
- Development Environment:
  - VS Code or PyCharm
  - Git for version control
  - Docker (for deployment)

## Risk Management Rules
1. Maximum position size: 2% of account per trade
2. Daily loss limit: 5% of account
3. Maximum drawdown: 15% of account
4. Stop-loss: ATR-based dynamic stops
5. Take-profit: 2:1 risk-reward ratio
6. Trading hours: Market-specific restrictions

## Immediate Next Steps
1. Implement training pipeline
   - Set up data loaders
   - Create training loop
   - Add validation metrics
2. Add feature engineering
   - Implement lagged features
   - Add rolling statistics
   - Create cross-indicator features
3. Set up model evaluation
   - Create performance metrics
   - Implement backtesting
   - Add risk management rules

## Notes
- Focus on robust feature engineering
- Implement proper model validation
- Maintain detailed experiment logs
- Regular model evaluation
- Document all decisions and changes
- Follow PEP 8 style guide
- Use type hints for better code maintainability
- Implement proper error handling
- Add comprehensive logging

## Project Todo List

## Completed Tasks
- [x] Set up project structure
- [x] Create basic ANN architecture
- [x] Implement data loading from CSV
- [x] Add visualization capabilities
- [x] Set up requirements.txt with necessary packages

## Current Progress
- Basic ANN implementation with 300 input neurons and 6 output neurons
- Data loading and preprocessing pipeline
- Model visualization using torchviz
- Project structure with separate directories for data, models, and visualization

## Next Steps
1. Data Preprocessing
   - [ ] Normalize input data
   - [ ] Split data into training and testing sets
   - [ ] Implement data augmentation if needed

2. Model Development
   - [ ] Implement training loop
   - [ ] Add validation metrics
   - [ ] Experiment with different architectures
   - [ ] Add hyperparameter tuning

3. Visualization
   - [ ] Add training progress visualization
   - [ ] Create model performance metrics plots
   - [ ] Implement real-time prediction visualization

4. Documentation
   - [ ] Add docstrings to all functions
   - [ ] Create API documentation
   - [ ] Add usage examples

5. Testing
   - [ ] Implement unit tests
   - [ ] Add integration tests
   - [ ] Performance testing

## Future Enhancements
- Implement different neural network architectures
- Add support for different data formats
- Create a web interface for model interaction
- Add automated model deployment pipeline

## Notes
- Current ANN architecture: 300 input neurons → 128 hidden neurons → 128 hidden neurons → 6 output neurons
- Using PyTorch as the deep learning framework
- Data visualization using torchviz and matplotlib
