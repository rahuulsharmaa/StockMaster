
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = warnings, 2 = errors, 3 = only critical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout,Bidirectional, MultiHeadAttention, Layer
from tensorflow.keras.optimizers import AdamW
# import os
import pickle
from datetime import datetime, timedelta
import hashlib
from functools import lru_cache
# Add to your imports
import pandas_ta as ta
# from tensorflow.keras.layers import Bidirectional, MultiHeadAttention, Layer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a global thread pool executor with limited workers to prevent resource exhaustion
thread_pool = ThreadPoolExecutor(max_workers=4)

# Directory for storing models
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Directory for caching data
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {}  # Cache for trained models
        self.data_cache = {}  # Cache for downloaded stock data
        
    def _get_cache_path(self, symbol, data_type='historical'):
        """Get path for caching data"""
        cache_key = f"{symbol.lower()}_{data_type}"
        hashed_key = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(CACHE_DIR, f'{hashed_key}.pkl')
        
    def _cache_data(self, symbol, data, data_type='historical'):
        """Cache data to disk"""
        cache_path = self._get_cache_path(symbol, data_type)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'timestamp': datetime.now()
                }, f)
            logger.debug(f"Cached {data_type} data for {symbol}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache {data_type} data for {symbol}: {e}")
            return False
            
    def _get_cached_data(self, symbol, max_age_hours=1, data_type='historical'):
        """Retrieve cached data if available and recent"""
        cache_path = self._get_cache_path(symbol, data_type)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                    
                # Check if cache is recent enough
                if (datetime.now() - cached['timestamp']) < timedelta(hours=max_age_hours):
                    logger.info(f"Using cached {data_type} data for {symbol}")
                    return cached['data']
            except Exception as e:
                logger.warning(f"Error loading cached {data_type} data for {symbol}: {e}")
                
        return None
    
    def _fetch_stock_data(self, symbol, period="3mo"):
        """Fetch stock data with caching"""
        # Check memory cache first
        if symbol in self.data_cache:
            cache_entry = self.data_cache[symbol]
            if (datetime.now() - cache_entry['timestamp']) < timedelta(hours=1):
                logger.info(f"Using memory-cached data for {symbol}")
                return cache_entry['data']
        
        # Check disk cache
        cached_data = self._get_cached_data(symbol)
        if cached_data is not None:
            # Update memory cache
            self.data_cache[symbol] = {
                'data': cached_data,
                'timestamp': datetime.now()
            }
            return cached_data
            
        # Fetch new data
        try:
            logger.info(f"Fetching new data for {symbol} for period {period}")
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if data.empty or len(data) < 30:
                logger.warning(f"Not enough data for {symbol}")
                return None
                
            # Cache the data
            self._cache_data(symbol, data)
            
            # Update memory cache
            self.data_cache[symbol] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    def _safe_float_conversion(self, array_value):
        """Safely convert NumPy array to float to avoid deprecated conversion warnings"""
        if hasattr(array_value, 'item'):
            return float(array_value.item())
        elif hasattr(array_value, '__len__') and len(array_value) == 1:
            return float(array_value[0])
        return float(array_value)        
    def _preprocess_data(self, data):
        """Enhanced preprocessing with additional technical indicators"""
        # Create a copy to avoid modifying the original
        processed_data = data.copy()
        
        # Calculate various moving averages
        processed_data['MA5'] = processed_data['Close'].rolling(window=5).mean()
        processed_data['MA10'] = processed_data['Close'].rolling(window=10).mean()
        processed_data['MA20'] = processed_data['Close'].rolling(window=20).mean()
        processed_data['MA50'] = processed_data['Close'].rolling(window=50).mean()
        
        # Calculate Exponential Moving Averages
        processed_data['EMA12'] = processed_data['Close'].ewm(span=12, adjust=False).mean()
        processed_data['EMA26'] = processed_data['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        processed_data['MACD'] = processed_data['EMA12'] - processed_data['EMA26']
        processed_data['MACD_signal'] = processed_data['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = processed_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        processed_data['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility (using Average True Range)
        processed_data['ATR'] = processed_data['High'] - processed_data['Low']
        processed_data['ATR'] = processed_data['ATR'].rolling(window=14).mean()
        
        # Calculate momentum
        processed_data['Momentum'] = processed_data['Close'].pct_change(periods=10) * 100
        
        # Fill NaN values with forward fill then backward fill - replace deprecated method
        processed_data = processed_data.ffill().bfill()
        
        # Extract metrics for the latest data point
        latest = processed_data.iloc[-1]
        
        metrics = {
            'current_price': latest['Close'],
            'ma5': latest['MA5'],
            'ma20': latest['MA20'],
            'ma50': latest['MA50'],
            'macd': latest['MACD'],
            'macd_signal': latest['MACD_signal'],
            'rsi': latest['RSI'],
            'momentum': latest['Momentum'],
            'volatility': latest['ATR'] / latest['Close'] * 100,  # As percentage of price
            'processed_data': processed_data  # Keep full processed dataframe for further analysis
        }
        
        return metrics
    
    
    # Price features
    # Add/modify these methods in your StockPredictor class

    def _prepare_lstm_data(self, data, sequence_length=60):
        """Prepare data for LSTM model with enhanced features."""
        # Create a copy of the data
        df = data.copy()
        
        # Calculate log returns instead of using raw prices
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Calculate lagged returns (important predictors)
        for lag in [1, 2, 3, 5, 10]:
            df[f'return_lag_{lag}'] = df['log_return'].shift(lag)
        
        # Create expanded technical indicators using pandas_ta
        # RSI with multiple timeframes
        df.ta.rsi(length=14, append=True)
        df.ta.rsi(length=7, append=True)
        df.ta.rsi(length=21, append=True)
        
        # MACD
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # Bollinger Bands
        df.ta.bbands(length=20, std=2, append=True)
        
        # Add rate of change
        df.ta.roc(length=10, append=True)
        df.ta.roc(length=20, append=True)
        
        # Add ADX (trend strength)
        df.ta.adx(length=14, append=True)
        
        # Add Stochastic Oscillator
        df.ta.stoch(k=14, d=3, append=True)
        
        # Add Advanced Volatility measures
        df.ta.atr(length=14, append=True)
        
        # Add market regime features (volatility regime)
        df['volatility_regime'] = df['log_return'].rolling(window=20).std()
        
        # Add price distance from moving averages (normalized)
        for window in [10, 20, 50]:
            ma = df['Close'].rolling(window=window).mean()
            df[f'dist_ma_{window}'] = (df['Close'] - ma) / ma
        
        # Fill NaN values
        df = df.dropna().reset_index(drop=True)
        
        # Extract target (next day return)
        y = df['log_return'].shift(-1).dropna().values
        
        # Drop the target column and any date/index columns
        X = df.drop(['log_return', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1, errors='ignore')
        if isinstance(X.index, pd.DatetimeIndex):
            # Add cyclical time features
            X['day_sin'] = np.sin(2 * np.pi * X.index.dayofweek / 7)
            X['day_cos'] = np.cos(2 * np.pi * X.index.dayofweek / 7)
            X['month_sin'] = np.sin(2 * np.pi * X.index.month / 12)
            X['month_cos'] = np.cos(2 * np.pi * X.index.month / 12)
        
        # Ensure all columns are numeric
        X = X.select_dtypes(include=[np.number])
        
        # Store column names for feature importance analysis
        self.feature_names = X.columns.tolist()
        
        # Scale features
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Scale target
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X_scaled) - 1):  # -1 because we shifted target by 1
            X_seq.append(X_scaled[i-sequence_length:i])
            y_seq.append(y_scaled[i])
            
        return np.array(X_seq), np.array(y_seq), df['Close'].values

    
    def _build_lstm_model(self, input_shape):
        """Build enhanced LSTM model with bidirectional layers and attention mechanism using Functional API"""
        # Set seeds for consistency
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Use Functional API instead of Sequential for more flexibility
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # First Bidirectional LSTM layer
        x = Bidirectional(LSTM(
            units=128,
            return_sequences=True, 
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0001, l2=0.0001),
            recurrent_regularizer=tf.keras.regularizers.l2(0.0001),
            recurrent_dropout=0.0
        ))(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = Dropout(0.4)(x)
        
        # Second Bidirectional LSTM layer
        x = Bidirectional(LSTM(
            units=96, 
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)
        ))(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Now we can use MultiHeadAttention correctly
        attn = MultiHeadAttention(
            num_heads=4, 
            key_dim=24
        )(x, x)  # Self-attention: query=key=value=x
        
        # Add residual connection and normalization
        x = tf.keras.layers.add([x, attn])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Third LSTM layer
        x = Bidirectional(LSTM(
            units=64, 
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)
        ))(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Dense layers
        x = Dense(
            units=48, 
            activation='swish',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)
        )(x)
        x = Dropout(0.3)(x)
        
        x = Dense(
            units=24, 
            activation='swish',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)
        )(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        outputs = Dense(units=1)(x)
        
        # Create model
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        
        # Compile with the same settings
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.001, 
            weight_decay=0.0001,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer, 
            loss='mse',
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(),
                tf.keras.metrics.MeanAbsoluteError()
            ]
        )
        
        return model
    def _get_model_path(self, symbol):
        """Get path for saving/loading model"""
        return os.path.join(MODEL_DIR, f'{symbol.lower()}_lstm_model.pkl')
    
    def _save_model(self, symbol, model, model_data):
        """Save model and related data to disk efficiently"""
        model_path = self._get_model_path(symbol)
        try:
            # Use more efficient serialization
            model_dict = {
                'model_config': model.get_config(),
                'weights': model.get_weights(),
                'feature_scaler': self.feature_scaler,
                'close_scaler': self.close_scaler,
                'last_price': model_data.get('last_price'),
                'last_updated': datetime.now(),
                'training_completed': True
            }
                
            with open(model_path, 'wb') as f:
                pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            logger.info(f"Model saved for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error saving model for {symbol}: {e}")
            return False
    def _load_model(self, symbol):
        """Load model from disk if available and not outdated"""
        model_path = self._get_model_path(symbol)
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_dict = pickle.load(f)
                
                # Check if model is recent (less than 12 hours old)
                if (datetime.now() - model_dict.get('last_updated', datetime(1970, 1, 1))) < timedelta(hours=12):
                    # Check for all required keys
                    required_keys = ['model_config', 'weights', 'feature_scaler', 'close_scaler', 'training_completed']
                    if all(key in model_dict for key in required_keys) and model_dict.get('training_completed', False):
                        # Reconstruct model from config
                        from tensorflow.keras.models import Sequential
                        model = Sequential.from_config(model_dict['model_config'])
                        model.set_weights(model_dict['weights'])
                        
                        # Use the same optimizer and loss as in build method
                        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                        model.compile(optimizer=optimizer, loss='mean_squared_error')
                        
                        # Restore scalers
                        self.feature_scaler = model_dict['feature_scaler']
                        self.close_scaler = model_dict['close_scaler']
                        
                        logger.info(f"Loaded existing model for {symbol}")
                        return model, model_dict
                    else:
                        logger.warning(f"Model for {symbol} is missing required data, retraining")
                else:
                    logger.info(f"Model for {symbol} is outdated, retraining")
            except Exception as e:
                logger.warning(f"Error loading model for {symbol}: {e}")
                    
        return None, {}
    def _get_technical_prediction(self, metrics):
        """Get technical analysis-based prediction"""
        # Base score starts at neutral
        base_score = 50
        signals = []
        
        # === Price trends ===
        if metrics['ma5'] > metrics['ma20']:
            base_score += 5
            signals.append("short-term uptrend")
        else:
            base_score -= 5
            signals.append("short-term downtrend")
            
        if metrics['ma20'] > metrics['ma50']:
            base_score += 5
            signals.append("medium-term uptrend")
        else:
            base_score -= 5
            signals.append("medium-term downtrend")
        
        # === MACD signal ===
        if metrics['macd'] > metrics['macd_signal']:
            base_score += 7
            signals.append("positive MACD crossover")
        else:
            base_score -= 7
            signals.append("negative MACD crossover")
        
        # === RSI signals ===
        if metrics['rsi'] < 30:
            base_score += 10  # Oversold
            signals.append("oversold (RSI)")
        elif metrics['rsi'] > 70:
            base_score -= 10  # Overbought
            signals.append("overbought (RSI)")
        
        # === Momentum ===
        momentum_impact = min(max(metrics['momentum'], -15), 15)
        base_score += momentum_impact
        
        if metrics['momentum'] > 3:
            signals.append("strong positive momentum")
        elif metrics['momentum'] < -3:
            signals.append("strong negative momentum")
        
        # === Volatility penalty ===
        # High volatility reduces confidence
        volatility_penalty = min(metrics['volatility'] * 0.5, 10)
        base_score -= volatility_penalty
        
        if metrics['volatility'] > 3:
            signals.append("high volatility")
        
        # Ensure score is within 0-100 range
        prediction_score = max(min(base_score, 100), 0)
        
        # Determine direction and confidence
        direction = "bullish" if prediction_score > 50 else "bearish"
        
        # Calculate confidence level
        if abs(prediction_score - 50) > 25:
            confidence = "High"
        elif abs(prediction_score - 50) > 10:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Predict price change based on score and current price
        # More dramatic score = more dramatic price change prediction
        price_change_pct = (prediction_score - 50) / 250  # Max ±20% prediction
        predicted_price = metrics['current_price'] * (1 + price_change_pct)
        
        return {
            "current_price": metrics['current_price'],
            "predicted_price": predicted_price,
            "direction": direction,
            "prediction_score": prediction_score,
            "confidence": confidence,
            "factors": ", ".join(signals)
        }
        
    def predict_without_ml(self, data, fundamentals):
        """Make prediction without using ML - improved using technical indicators"""
        # Process data with enhanced metrics
        metrics = self._preprocess_data(data)
        
        # Get technical prediction
        prediction = self._get_technical_prediction(metrics)
        
        # Add fundamental factors if available
        if fundamentals:
            fundamental_factors = self._analyze_fundamentals(fundamentals, prediction['direction'])
            if fundamental_factors:
                prediction['factors'] += f", {fundamental_factors}"
        
        prediction['success'] = True
        return prediction
    
    def _analyze_fundamentals(self, fundamentals, direction):
        """Analyze fundamental data for factors"""
        factors = []
        
        # P/E ratio analysis
        if fundamentals.get("pe_ratio"):
            pe = fundamentals.get("pe_ratio")
            if pe and pe < 15 and direction == "bullish":
                factors.append("attractive P/E ratio")
            elif pe and pe > 30 and direction == "bearish":
                factors.append("high P/E ratio")
        
        # Dividend yield
        if fundamentals.get("dividend_yield", 0) > 0.03:
            factors.append("strong dividend yield")
            
        # Market cap
        if fundamentals.get("market_cap", 0) > 100_000_000_000:  # $100B+
            factors.append("large market cap")
            
        # Profit margin
        if fundamentals.get("profit_margin", 0) > 0.2:  # 20%+
            factors.append("strong profit margins")
        elif fundamentals.get("profit_margin", 0) < 0.05:  # <5%
            factors.append("thin profit margins")
            
        # Debt to equity
        if fundamentals.get("debt_to_equity", 0) > 2:
            factors.append("high debt levels")
            
        return ", ".join(factors)
    
    @lru_cache(maxsize=32)
    def _get_fundamentals(self, symbol):
        """Get fundamental data for the stock with caching"""
        # Check disk cache first
        cached_data = self._get_cached_data(symbol, max_age_hours=24, data_type='fundamentals')
        if cached_data is not None:
            return cached_data
            
        try:
            # Get basic info
            stock = yf.Ticker(symbol)
            info = stock.info
            
            fundamentals = {
                "pe_ratio": info.get("trailingPE", None),
                "forward_pe": info.get("forwardPE", None),
                "peg_ratio": info.get("pegRatio", None),
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", 0),
                "market_cap": info.get("marketCap", 0),
                "profit_margin": info.get("profitMargins", 0),
                "return_on_equity": info.get("returnOnEquity", 0),
                "debt_to_equity": info.get("debtToEquity", 0)
            }
            
            # Cache the data
            self._cache_data(symbol, fundamentals, data_type='fundamentals')
            
            return fundamentals
        except Exception as e:
            logger.warning(f"Error fetching fundamentals for {symbol}: {e}")
            return {}
    
    def predict_with_lstm(self, symbol, data, fundamentals):
        """Make prediction using LSTM model with enhanced stability"""
        try:
            # Set seeds for consistent results
            np.random.seed(42)
            tf.random.set_seed(42)
            
            # Check if we already have a trained model
            if symbol in self.models:
                model = self.models[symbol]['model']
                model_data = self.models[symbol]['data']
                logger.info(f"Using cached model for {symbol}")
            else:
                # Try to load model from disk
                model, model_data = self._load_model(symbol)
                
                # If no model exists or it's outdated, create and train a new one
                if model is None:
                    logger.info(f"Training new LSTM model for {symbol}")
                    
                    # Prepare data for LSTM with enhanced features
                    X_data, y_data, scaled_data = self._prepare_lstm_data(data)
                    
                    if len(X_data) < 10:  # Not enough data
                        logger.warning(f"Not enough data for LSTM model for {symbol}")
                        return self.predict_without_ml(data, fundamentals)
                    
                    # Build model with consistent configuration
                    model = self._build_lstm_model((X_data.shape[1], X_data.shape[2]))
                    
                    # Train model with fixed random seed and early stopping
                    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='loss', patience=10, restore_best_weights=True
                    )
                    
                    callbacks = [
                        tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=15, restore_best_weights=True
                            ),
                        tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001
                        )
                    ]   
                    # Use fixed batch size and epochs for reproducibility
                    model.fit(
                        X_data, y_data,
                        epochs=100,
                        batch_size=32, 
                        validation_split=0.2,
                        callbacks=callbacks,
                        verbose=0,
                        shuffle=False  # Avoid random shuffling for reproducibility
                    )
                    
                    # Save model data
                    model_data = {
                        'last_price': data['Close'].iloc[-1],
                        'last_updated': datetime.now()
                    }
                    
                    # Cache and save the model
                    self.models[symbol] = {'model': model, 'data': model_data}
                    self._save_model(symbol, model, model_data)
                else:
                    # Cache loaded model
                    self.models[symbol] = {'model': model, 'data': model_data}
            
            # Process data with enhanced features for prediction
            processed_data = self._preprocess_data(data)['processed_data']
            
            # Prepare features for prediction (same as in training)
            features_df = pd.DataFrame()
            features_df['close'] = processed_data['Close']
            features_df['high_low_ratio'] = processed_data['High'] / processed_data['Low']
            features_df['close_open_ratio'] = processed_data['Close'] / processed_data['Open']
            features_df['ma5'] = processed_data['MA5']
            features_df['ma20'] = processed_data['MA20']
            features_df['volatility'] = processed_data['Close'].pct_change().rolling(window=5).std()
            features_df['momentum'] = processed_data['Close'].pct_change(periods=5)
            
            if 'Volume' in processed_data.columns:
                features_df['volume_change'] = processed_data['Volume'].pct_change()
                features_df['price_volume_ratio'] = processed_data['Close'] * processed_data['Volume']
                
            # Fill NaN values - replace deprecated method
            features_df = features_df.ffill().bfill().fillna(0)
            
            # Use the feature_scaler that was saved during training
            if not hasattr(self, 'feature_scaler') or self.feature_scaler is None:
                # If feature_scaler doesn't exist, refit it (this should rarely happen if save/load works correctly)
                logger.warning(f"feature_scaler not found for {symbol}, refitting")
                self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
                self.feature_scaler.fit(features_df)
            
            # Scale features using the same scaler used during training
            scaled_features = self.feature_scaler.transform(features_df)
            
            # Create input sequence for prediction (last sequence_length points)
            sequence_length = 60
            x_test = scaled_features[-sequence_length:].reshape(1, sequence_length, -1)
            
            # Make prediction with multiple runs for stability
            predictions = []
            for _ in range(5):  # Run prediction 5 times
                predicted_scaled = model.predict(x_test, verbose=0)[0][0]
                predictions.append(predicted_scaled)
                
            # Average the predictions for stability
            predicted_scaled_avg = sum(predictions) / len(predictions)
            
            # Transform back to original scale using the close_scaler
            if not hasattr(self, 'close_scaler') or self.close_scaler is None:
                # If close_scaler doesn't exist, refit it
                logger.warning(f"close_scaler not found for {symbol}, refitting")
                self.close_scaler = MinMaxScaler(feature_range=(0, 1))
                self.close_scaler.fit(data['Close'].values.reshape(-1, 1))
                
            predicted_price = self.close_scaler.inverse_transform([[predicted_scaled_avg]])[0][0]
            
            current_price = data['Close'].iloc[-1]
            price_diff_pct = ((predicted_price - current_price) / current_price) * 100
            
            # Determine direction and confidence
            direction = "bullish" if predicted_price > current_price else "bearish"
            
            # Calculate prediction score (50 = neutral, 0-100 scale)
            base_score = 50
            price_impact = min(max(price_diff_pct * 2, -30), 30)  # Cap at ±30 points
            
            prediction_score = base_score + price_impact if direction == "bullish" else base_score - abs(price_impact)
            prediction_score = max(min(prediction_score, 100), 0)  # Ensure 0-100 range
            
            # Determine confidence level
            if abs(prediction_score - 50) > 25:
                confidence = "High"
            elif abs(prediction_score - 50) > 10:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            # Get technical analysis factors
            metrics = self._preprocess_data(data)
            technical_prediction = self._get_technical_prediction(metrics)
            
            # Combine technical and ML factors
            ml_factor = "LSTM prediction"
            factors = technical_prediction['factors']
            
            # Add fundamental factors if available
            if fundamentals:
                fundamental_factors = self._analyze_fundamentals(fundamentals, direction)
                if fundamental_factors:
                    factors += f", {fundamental_factors}"
            
            # Add ML as primary factor
            factors = f"{ml_factor} based on {factors}"
            
            return {
                "success": True,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "direction": direction,
                "prediction_score": prediction_score,
                "confidence": confidence,
                "factors": factors
            }
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction for {symbol}: {str(e)}")
            # Log traceback for detailed debugging
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to non-ML prediction
            return self.predict_without_ml(data, fundamentals)
    
    def train_and_predict(self, symbol):
        """Get prediction for a stock symbol with enhanced stability and performance"""
        try:
            # Validate symbol input
            if not symbol or not isinstance(symbol, str):
                return {
                    "success": False,
                    "error": "Invalid stock symbol"
                }
                
            # Standardize symbol format
            symbol = symbol.upper().strip()
            
            # Download data for the past 3 months (longer period for better training)
            data = self._fetch_stock_data(symbol, period="6mo")
            
            if data is None or data.empty or len(data) < 30:
                logger.warning(f"Not enough data for {symbol}")
                return {
                    "success": False,
                    "error": "Not enough historical data available"
                }
            
            # Get fundamentals with caching
            fundamentals = self._get_fundamentals(symbol)
            
            # Use ML-based prediction
            logger.info(f"Generating LSTM prediction for {symbol}")
            try:
                prediction = self.predict_with_lstm(symbol, data, fundamentals)
            except Exception as e:
                logger.error(f"LSTM prediction failed for {symbol}: {e}")
                # Fallback to technical analysis
                prediction = self.predict_without_ml(data, fundamentals)
                
            # Format prediction values for consistency
            prediction["current_price"] = round(self._safe_float_conversion(prediction["current_price"]), 2)
            prediction["predicted_price"] = round(self._safe_float_conversion(prediction["predicted_price"]), 2)
            prediction["prediction_score"] = round(self._safe_float_conversion(prediction["prediction_score"]), 1)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to generate prediction: {str(e)}"
            }
            
    def evaluate_model(self, symbol, test_size=0.2):
        """Evaluate model accuracy and check for overfitting with improved stability"""
        logger.info(f"Evaluating model for {symbol}...")
        
        try:
            # Enable mixed precision training for faster computation
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
            except AttributeError:
                try:
                    mixed_precision = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
                    tf.keras.mixed_precision.experimental.set_policy(mixed_precision)
                except:
                    logger.warning("Mixed precision not available, continuing with default precision")
            
            # Fetch less data for faster evaluation (6 months instead of 2 years)
            data = self._fetch_stock_data(symbol, period="6mo")
            
            # Reduce minimum data requirement from 100 to 50
            if data is None or data.empty or len(data) < 50:
                logger.warning(f"Not enough data for evaluation of {symbol}")
                return {
                    "success": False,
                    "error": "Not enough historical data for proper evaluation"
                }
            
            # Prepare data with shorter sequence length (30 instead of 90)
            X_data, y_data, close_prices = self._prepare_lstm_data(data, sequence_length=30)
            
            # Add progress logging
            logger.info(f"Data prepared for {symbol}: {len(X_data)} sequences, feature shape: {X_data.shape}")
            
            # Initialize the close_scaler if it doesn't exist
            if not hasattr(self, 'close_scaler') or self.close_scaler is None:
                self.close_scaler = MinMaxScaler(feature_range=(0, 1))
                close_values = data['Close'].values.reshape(-1, 1)
                self.close_scaler.fit(close_values)
            
            # Split data into training and testing sets
            split_idx = int(len(X_data) * (1 - test_size))
            X_train, X_test = X_data[:split_idx], X_data[split_idx:]
            y_train, y_test = y_data[:split_idx], y_data[split_idx:]
            
            # Pre-calculate the validation data to avoid repeated splitting
            val_split_idx = int(len(X_train) * 0.8)
            X_train_fit, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
            y_train_fit, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
            
            # Run single model instead of ensemble for faster evaluation
            num_models = 1  # Reduced from 2 to 1 for faster evaluation
            models = []
            predictions = []
            
            for i in range(num_models):
                # Set different seeds for each model
                seed = 42 + i
                tf.random.set_seed(seed)
                np.random.seed(seed)
                
                # Build and train model
                model = self._build_lstm_model((X_train.shape[1], X_train.shape[2]))
                
                # Learning rate scheduler for better convergence
                lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5,
                    patience=2,  # Reduced from 5
                    min_lr=0.0001
                )
                
                # Early stopping to prevent overfitting
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=5,  # Reduced from 15
                    restore_best_weights=True
                )
                
                # Train with validation split - optimized parameters
                history = model.fit(
                    X_train_fit, y_train_fit,
                    validation_data=(X_val, y_val),
                    epochs=20,  # Reduced from 50 for faster results
                    batch_size=128,  # Keep this for faster processing
                    callbacks=[early_stopping, lr_scheduler],
                    verbose=0,
                    shuffle=False  # For reproducibility
                )
                train_loss = model.evaluate(X_train, y_train, verbose=0)
                test_loss = model.evaluate(X_test, y_test, verbose=0)
                if isinstance(train_loss, (list, tuple)):
                    train_loss = train_loss[0]
                if isinstance(test_loss, (list, tuple)):
                    test_loss = test_loss[0]
                # Store model and make predictions
                models.append({
                    'model': model,
                    'history': history,
                    'train_loss': train_loss,
                    'test_loss': test_loss
                })
                
                # Get prediction in one batch
                pred = model.predict(X_test, batch_size=128, verbose=0).flatten()
                predictions.append(pred)
            
            # For single model, just use the first prediction directly
            y_pred_ensemble = predictions[0]
            
            # Calculate directional accuracy based on log returns
            actual_direction = np.sign(y_test)
            pred_direction = np.sign(y_pred_ensemble)
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            # Calculate metrics for the ensemble
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Convert log returns back to prices for error metrics
            last_price_train = close_prices[split_idx-1]
            
            # Reconstruct price series from log returns
            y_test_prices = [last_price_train]
            for log_return in y_test:
                # Inverse transform the log return
                clipped_return = np.clip(log_return, -1, 1)  # Limit returns to ±100%
                actual_return = self.close_scaler.inverse_transform([[clipped_return]])[0][0]
                actual_return = np.clip(actual_return, -50, 50) 
                 # Ensure return doesn't cause overflow
                try:
                    next_price = y_test_prices[-1] * np.exp(actual_return)
                except OverflowError:
                    logger.warning("Overflow in np.exp(actual_return); setting next_price = prev_price")
                    next_price = y_test_prices[-1]
                y_test_prices.append(next_price)
            y_test_prices = y_test_prices[1:]  # Remove the initial price
            
            # Do the same for predictions
            y_pred_prices = [last_price_train]
            for log_return in y_pred_ensemble:
                # Inverse transform the log return
                clipped_return = np.clip(log_return, -1, 1)  # Limit returns to ±100%
                pred_return = self.close_scaler.inverse_transform([[clipped_return]])[0][0]
                pred_return = np.clip(pred_return, -50, 50)
                # Ensure return doesn't cause overflow
                try:
                    next_price = y_pred_prices[-1] * np.exp(pred_return)
                except OverflowError:
                    logger.warning("Overflow in np.exp(pred_return); setting next_price = prev_price")
                    next_price = y_pred_prices[-1]
                y_pred_prices.append(next_price)
            y_pred_prices = y_pred_prices[1:]  # Remove the initial price
            
            # Calculate error metrics on prices - faster by using numpy arrays directly
            y_test_arr = np.array(y_test_prices)
            y_pred_arr = np.array(y_pred_prices)
            
            rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred_arr))
            mae = mean_absolute_error(y_test_arr, y_pred_arr)
            mape = np.mean(np.abs((y_test_arr - y_pred_arr) / y_test_arr)) * 100
            r2 = r2_score(y_test_arr, y_pred_arr)
            
            # Classify as binary problem (up/down)
            y_true_bin = (y_test > 0).astype(int)
            y_pred_bin = (y_pred_ensemble > 0).astype(int)
            
            # Calculate classification metrics
            accuracy = accuracy_score(y_true_bin, y_pred_bin)
            precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            
            # Calculate average model metrics (just one model now)
            avg_train_loss = models[0]['train_loss']
            avg_test_loss = models[0]['test_loss']
            
            # Check for overfitting (using less strict threshold due to ensemble)
            is_overfitting = avg_train_loss < avg_test_loss * 0.8
            
            # Check for underfitting
            is_underfitting = r2 < 0.5 or directional_accuracy < 55
            
            # Print comprehensive results
            print("\n" + "="*60)
            print(f"MODEL EVALUATION RESULTS FOR {symbol}")
            print("="*60)
            print(f"Avg Training Loss: {avg_train_loss:.6f}")
            print(f"Avg Testing Loss: {avg_test_loss:.6f}")
            print(f"RMSE: ${rmse:.2f}")
            print(f"MAE: ${mae:.2f}")
            print(f"MAPE: {mape:.2f}%")
            print(f"R² Score: {r2:.4f}")
            print(f"Directional Accuracy: {directional_accuracy:.2f}%")
            print(f"Binary Classification Accuracy: {accuracy * 100:.2f}%")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("-"*60)
            
            if is_overfitting:
                print("MODEL STATUS: OVERFITTING - Training loss much lower than testing loss")
                print("RECOMMENDATION: Increase dropout rate, add more regularization, collect more data")
            elif is_underfitting:
                print("MODEL STATUS: UNDERFITTING - Model isn't capturing patterns well")
                print("RECOMMENDATION: Add more features, try different architectures, tune hyperparameters")
            else:
                print("MODEL STATUS: GOOD FIT - Model appears to be properly trained")
            
            print("="*60)
            
            return {
                "success": True,
                "train_loss": avg_train_loss,
                "test_loss": avg_test_loss,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "r2": r2,
                "directional_accuracy": directional_accuracy,
                "binary_accuracy": accuracy * 100,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "is_overfitting": is_overfitting,
                "is_underfitting": is_underfitting
            }
            
        except Exception as e:
            logger.error(f"Error in model evaluation for {symbol}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Evaluation failed: {str(e)}"
            }
    def _calculate_directional_accuracy(self, y_true, y_pred):
        """Calculate the accuracy of predicting the direction of price movement"""
        # Convert to direction (1 for up, -1 for down)
        y_true_dir = np.sign(y_true)
        y_pred_dir = np.sign(y_pred)
        
        # Calculate directional accuracy
        correct_dir = (y_true_dir == y_pred_dir)
        return np.mean(correct_dir) * 100

    

# Initialize the predictor as a singleton
stock_predictor = StockPredictor()

async def get_stock_prediction(symbol):
    """Get stock prediction for the given symbol with better error handling"""
    try:
        # Run the CPU-intensive work in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            thread_pool, 
            stock_predictor.train_and_predict, 
            symbol
        )
        
        return result
    except Exception as e:
        logger.error(f"Unexpected error in get_stock_prediction for {symbol}: {str(e)}")
        return {
            "success": False,
            "error": "An unexpected error occurred while processing your request"
        }
def evaluate_stock_model(symbol='AAPL'):
    """Command-line function to evaluate the ensemble model for a given stock"""
    predictor = StockPredictor()
    results = predictor.evaluate_model(symbol)
    
    if not results.get('success', False):
        print(f"Evaluation failed: {results.get('error', 'Unknown error')}")
        return
    
    # Print performance summary
    print("\nPERFORMANCE SUMMARY:")
    print(f"The ensemble model for {symbol} has:")
    print(f"- Directional Accuracy: {results['directional_accuracy']:.2f}% (goal: >60%)")
    print(f"- Average Price Error: ${results['mae']:.2f}")
    print(f"- R² Score: {results['r2']:.4f} (goal: >0.6)")
    
    # Print stability assessment
    if abs(results['train_loss'] - results['test_loss']) < 0.01:
        print("\nSTABILITY ASSESSMENT: Excellent - minimal gap between training and testing")
    elif abs(results['train_loss'] - results['test_loss']) < 0.05:
        print("\nSTABILITY ASSESSMENT: Good - reasonable gap between training and testing")
    else:
        print("\nSTABILITY ASSESSMENT: Poor - large gap between training and testing")
    
    # Improvement suggestions based on actual metrics
    print("\nIMPROVEMENT RECOMMENDATIONS:")
    if results['directional_accuracy'] < 60:
        print("- Focus on directional prediction by using classification for direction + regression for magnitude")
    if results['mape'] > 3.0:
        print("- Include more market sentiment features (news sentiment, social media indicators)")
    if results['r2'] < 0.6:
        print("- Try using a different model architecture (GRU or Transformer)")
    if results['is_overfitting']:
        print("- Use more aggressive regularization and shorter training periods")
    
    return results

# If this file is run directly, evaluate a default stock
if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    print(f"Evaluating stock prediction model for {symbol}...")
    evaluate_stock_model(symbol)