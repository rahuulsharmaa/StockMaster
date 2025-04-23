"""Stock prediction module using machine learning and technical analysis."""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf



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

# Add this to speed up data processing
FAST_EVAL = False  # Flag to enable fast evaluation mode

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {}  # Cache for trained models
        self.data_cache = {}  # Cache for downloaded stock data
        self.close_scaler = None
        self.feature_scaler = None
        self.feature_names = []

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
            logger.debug("Cached %s data for %s", data_type, symbol)
            return True
        except Exception as e:
            logger.warning("Failed to cache %s data for %s: %s", data_type, symbol, e)
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
                    logger.info("Using cached %s data for %s", data_type, symbol)
                    return cached['data']
            except Exception as e:
                logger.warning("Error loading cached %s data for %s: %s", data_type, symbol, e)

        return None

    def _fetch_stock_data(self, symbol, period="3mo"):
        """Fetch stock data with caching"""
        # Check memory cache first
        if symbol in self.data_cache:
            cache_entry = self.data_cache[symbol]
            if (datetime.now() - cache_entry['timestamp']) < timedelta(hours=1):
                logger.info("Using memory-cached data for %s", symbol)
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
            logger.info("Fetching new data for %s for period %s", symbol, period)
            stock = yf.Ticker(symbol)
            # Set auto_adjust explicitly to handle yfinance API changes
            data = stock.history(period=period, auto_adjust=True)

            if data.empty or len(data) < 30:
                logger.warning("Not enough data for %s", symbol)
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
            logger.error("Error fetching data for %s: %s", symbol, e)
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
        # FIXED: Replace deprecated fillna(method='ffill') with ffill()
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

    def _prepare_lstm_data(self, data, sequence_length=None, return_scalers=False):
        """Prepare data for LSTM model with enhanced features for better price prediction."""
        # Use default sequence length based on evaluation mode
        if sequence_length is None:
            sequence_length = 30 if FAST_EVAL else 60

        # Create a copy of the data
        df = data.copy()

        # Handle missing columns (sometimes yfinance returns different column sets)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan

        # Fill missing values with forward-fill then backward-fill
        df = df.ffill().bfill()

        # Enhanced price features - multiple timeframes for better prediction
        # Calculate log returns (more stable than raw prices)
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['log_return_ma5'] = df['log_return'].rolling(window=5).mean()

        # Add normalized price levels (normalized to recent history)
        for window in [10, 20, 50, 100]:
            df[f'norm_close_{window}'] = df['Close'] / df['Close'].rolling(window=window).mean() - 1

        # Add price volatility at different timeframes
        for window in [5, 10, 20, 50]:
            df[f'volatility_{window}'] = df['log_return'].rolling(window=window).std()

        # Add log returns for High, Low, Open for better price range prediction
        df['high_log_return'] = np.log(df['High'] / df['High'].shift(1))
        df['low_log_return'] = np.log(df['Low'] / df['Low'].shift(1))
        df['open_log_return'] = np.log(df['Open'] / df['Open'].shift(1))

        # Add lagged returns (important predictors) at multiple timeframes
        for lag in [1, 2, 3, 5, 10, 15, 20]:
            df[f'return_lag_{lag}'] = df['log_return'].shift(lag)

        # Calculate moving averages at multiple timeframes
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
            # Normalized distance from moving average (better than raw distances)
            df[f'ma_dist_{window}'] = df['Close'] / df[f'ma_{window}'] - 1 * 100

        # Add exponential moving averages (respond faster to recent price changes)
        for span in [5, 12, 26, 50]:
            df[f'ema_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
            # Normalized distance from EMA
            df[f'ema_dist_{span}'] = df['Close'] / df[f'ema_{span}'] - 1 * 100

        # Moving average crossovers (important signals)
        # Short-term crossovers
        df['ma_5_10_cross'] = df['ma_5'] > df['ma_10'].astype(int)
        df['ma_10_20_cross'] = df['ma_10'] > df['ma_20'].astype(int)
        # Medium-term crossovers
        df['ma_20_50_cross'] = df['ma_20'] > df['ma_50'].astype(int)
        # Long-term crossovers (golden/death cross)
        df['ma_50_200_cross'] = df['ma_50'] > df['ma_200'].astype(int)

        # Technical indicators with error handling
        try:
            # Enhanced RSI with multiple timeframes
            for length in [7, 14, 21]:
                df[f'rsi_{length}'] = df.ta.rsi(length=length)

            # MACD (used by many traders)
            macd = df.ta.macd(fast=12, slow=26, signal=9)
            if isinstance(macd, pd.DataFrame):
                for col in macd.columns:
                    df[col] = macd[col]

            # Bollinger Bands
            bbands = df.ta.bbands(length=20, std=2)
            if isinstance(bbands, pd.DataFrame):
                for col in bbands.columns:
                    df[col] = bbands[col]

                # Add BB width and position (better signals than raw values)
                if all(col in df.columns for col in ['BBU_20_2.0', 'BBL_20_2.0']):
                    df['bb_width'] = df['BBU_20_2.0'] - df['BBL_20_2.0'] / df['BBM_20_2.0']
                    df['bb_position'] = df['Close'] - df['BBL_20_2.0'] / (df['BBU_20_2.0'] - df['BBL_20_2.0'])

            # Rate of change (momentum indicators)
            for length in [5, 10, 20]:
                df[f'roc_{length}'] = df.ta.roc(length=length)

            # Add ADX (trend strength)
            adx = df.ta.adx(length=14)
            if isinstance(adx, pd.DataFrame):
                for col in adx.columns:
                    df[col] = adx[col]

            # Add Stochastic Oscillator (overbought/oversold indicator)
            stoch = df.ta.stoch(k=14, d=3)
            if isinstance(stoch, pd.DataFrame):
                for col in stoch.columns:
                    df[col] = stoch[col]
        except Exception as e:
            logger.warning("Error calculating technical indicators: %s. Using simplified feature set.", e)
            # Simple RSI calculation as fallback
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

        # Enhanced Volume features for better price prediction
        if 'Volume' in df.columns and not df['Volume'].isna().all():
            # Basic volume normalization
            df['volume_ma10'] = df['Volume'].rolling(window=10).mean()
            df['rel_volume'] = df['Volume'] / df['volume_ma10']

            # Volume changes
            df['volume_change'] = df['Volume'].pct_change()
            df['volume_change_ma5'] = df['volume_change'].rolling(window=5).mean()

            # Money Flow features
            df['money_flow'] = df['Close'] * df['Volume']
            df['money_flow_ma10'] = df['money_flow'].rolling(window=10).mean()

            # Price-Volume relationship features (often predictive)
            df['close_up'] = ((df['Close'] > df['Close'].shift(1))).astype(int)
            df['up_volume'] = df['Volume'] * df['close_up']
            df['down_volume'] = df['Volume'] * (1 - df['close_up'])

            # On-Balance Volume with smoothing
            df['obv'] = df['close_up'] * 2 - 1 * df['Volume']
            df['obv'] = df['obv'].cumsum()
            df['obv_ma10'] = df['obv'].rolling(window=10).mean()

        # Market regime features (aid in prediction during different market conditions)
        df['volatility_regime'] = df['log_return'].rolling(window=20).std()
        df['trend_strength'] = abs(df['ma_50'] / df['ma_200'] - 1) * 100

        # Add time-based features (markets often have time-based patterns)
        if isinstance(df.index, pd.DatetimeIndex):
            # Day of week (cyclical encoding)
            df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

            # Month features (seasonal patterns)
            df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

        # Fill NaN values with appropriate methods for different feature types
        # First use forward fill for most features
        df = df.ffill()

        # For features where 0 is more appropriate than ffill
        zero_fill_columns = [col for col in df.columns if 'change' in col or 'roc' in col]
        for col in zero_fill_columns:
            df[col] = df[col].fillna(0)

        # For features where mean is more appropriate
        mean_fill_columns = [col for col in df.columns if 'rsi' in col or 'position' in col]
        for col in mean_fill_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())

        # Final cleanup - any remaining NaNs with 0
        df = df.fillna(0)

        # Extract target (next day return)
        if len(df) <= 1:
            logger.error("Not enough data points to create features and targets")
            if return_scalers:
                return np.array([]), np.array([]), np.array([]), {}
            return np.array([]), np.array([]), np.array([])

        # Target is shifted log return
        y = df['log_return'].shift(-1).dropna().values

        # Drop columns that shouldn't be features
        drop_cols = ['log_return', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_df = df.drop([col for col in drop_cols if col in df.columns], axis=1)

        # Drop the last row since target is shifted
        feature_df = feature_df.iloc[:-1]

        # Ensure all columns are numeric
        feature_df = feature_df.select_dtypes(include=[np.number])

        # Store feature names for later use
        self.feature_names = feature_df.columns.tolist()
        logger.info("Using %s features for prediction", len(self.feature_names))

        # Scale features with improved scaler settings
        self.feature_scaler = MinMaxScaler(feature_range=(-1,
    1))  # Wider range for better gradients
        x_scaled = self.feature_scaler.fit_transform(feature_df)

        # Scale target (log returns) - better to use a wider range for prediction
        self.close_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_scaled = self.close_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # Create sequences with overlap for better training
        x_seq, y_seq = [], []
        for i in range(sequence_length, len(x_scaled)):
            x_seq.append(x_scaled[i-sequence_length:i])
            y_seq.append(y_scaled[i-sequence_length])

        # Convert to numpy arrays
        x_seq = np.array(x_seq)
        y_seq = np.array(y_seq)

        logger.info("Created %s sequences of length %s", len(x_seq), sequence_length)

        if return_scalers:
            return x_seq, y_seq, df['Close'].values, {
                'close_scaler': self.close_scaler,
                'feature_scaler': self.feature_scaler
            }
        else:
            return x_seq, y_seq, df['Close'].values

    def _build_lstm_model(self, input_shape):
        """Build robust LSTM model focused on better price prediction"""
        # Set seeds for consistency
        tf.random.set_seed(42)
        np.random.seed(42)

        # Use Functional API for all models to allow for attention layers
        inputs = tf.keras.layers.Input(shape=input_shape)

        # First LSTM layer
        x = Bidirectional(LSTM(
            64,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.0
        ))(inputs)

        # Add layer normalization
        x = tf.keras.layers.LayerNormalization()(x)

        # Second LSTM layer
        x = Bidirectional(LSTM(
            32,
            return_sequences=True,
            dropout=0.2
        ))(x)

        # Add layer normalization
        x = tf.keras.layers.LayerNormalization()(x)

        # Final LSTM layer
        x = LSTM(
            32,
            return_sequences=False,
            dropout=0.1
        )(x)

        # Dense layers
        x = tf.keras.layers.Dense(
            32,
            activation='swish',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.LayerNormalization()(x)

        # Final dense layer
        outputs = tf.keras.layers.Dense(1)(x)

        # Create model
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        # Use Adam optimizer with sensible defaults
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0
        )

        # Compile with MSE loss for better price prediction
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
                'feature_names': self.feature_names,
                'last_price': model_data.get('last_price'),
                'last_updated': datetime.now(),
                'training_completed': True,
                'model_type': 'functional' if not isinstance(model, Sequential) else 'sequential',
    # Flag to indicate the model type
                'input_shape': model_data.get('input_shape'),
    # Save the input shape for compatibility checking
                'sequence_length': model_data.get('sequence_length',
    30 if FAST_EVAL else 60)  # Save the sequence length used
            }

            with open(model_path, 'wb') as f:
                pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info("Model saved for %s", symbol)
            return True
        except Exception as e:
            logger.error("Error saving model for %s: %s", symbol, e)
            return False
    def _load_model(self, symbol):
        """Load model from disk if available and not outdated"""
        model_path = self._get_model_path(symbol)

        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_dict = pickle.load(f)

                # Check if model is recent (less than 12 hours old)
                if (datetime.now() - model_dict.get('last_updated', datetime(1970, 1,
    1))) < timedelta(hours=12):
                    # Check for all required keys
                    required_keys = ['model_config', 'weights', 'feature_scaler', 'close_scaler',
    'feature_names', 'training_completed']
                    if all(key in model_dict for key in required_keys) and model_dict.get('training_completed', False):
                        # Check model type and reconstruct appropriately
                        if model_dict.get('model_type') == 'functional':
                            # For Functional API models
                            model = tf.keras.models.Model.from_config(model_dict['model_config'])
                        else:
                            # For Sequential models (legacy support)
                            model = tf.keras.models.Sequential.from_config(model_dict['model_config'])

                        model.set_weights(model_dict['weights'])

                        # Use the same optimizer and loss as in build method
                        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)
                        model.compile(
                            optimizer=optimizer,
                            loss='mse',
                            metrics=[
                                tf.keras.metrics.RootMeanSquaredError(),
                                tf.keras.metrics.MeanAbsoluteError()
                            ]
                        )

                        # Restore scalers and feature names
                        self.feature_scaler = model_dict['feature_scaler']
                        self.close_scaler = model_dict['close_scaler']
                        self.feature_names = model_dict['feature_names']

                        # Extract input shape info if available
                        model_data = {
                            'last_price': model_dict.get('last_price'),
                            'last_updated': model_dict.get('last_updated'),
                            'input_shape': model_dict.get('input_shape'),
                            'sequence_length': model_dict.get('sequence_length',
    60)  # Default to 60 if not saved
                        }

                        logger.info("Loaded existing model for %s", symbol)
                        return model, model_data
                    logger.warning("Model for %s is missing required data, retraining", symbol)
                else:
                    logger.info("Model for %s is outdated, retraining", symbol)
            except Exception as e:
                logger.warning("Error loading model for %s: %s", symbol, e)

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
        price_change_pct = prediction_score - 50 / 250  # Max ±20% prediction
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
            logger.warning("Error fetching fundamentals for %s: %s", symbol, e)
            return {}

    def predict_with_lstm(self, symbol, data, fundamentals):
        """Make prediction using enhanced LSTM model with improved features and robustness"""
        try:
            # Set seeds for consistent results
            np.random.seed(42)
            tf.random.set_seed(42)

            # Check if we already have a trained model
            if symbol in self.models:
                model = self.models[symbol]['model']
                model_data = self.models[symbol]['data']
                logger.info("Using cached model for %s", symbol)
            else:
                # Try to load model from disk
                model, model_data = self._load_model(symbol)

                # If no model exists or it's outdated, create and train a new one
                if model is None:
                    logger.info("Training new LSTM model for %s", symbol)

                    # Get the sequence length based on current mode
                    sequence_length = 30 if FAST_EVAL else 60

                    # Prepare data for LSTM with enhanced features
                    x_data, y_data, scaled_data, scalers = self._prepare_lstm_data(data,
    sequence_length=sequence_length, return_scalers=True)

                    if len(x_data) < 10:  # Not enough data
                        logger.warning("Not enough data for LSTM model for %s", symbol)
                        return self.predict_without_ml(data, fundamentals)

                    # Store input shape for later compatibility checking
                    input_shape = (x_data.shape[1], x_data.shape[2])

                    # Build model with consistent configuration
                    model = self._build_lstm_model(input_shape)

                    # Create train/validation split
                    val_split_idx = int(len(x_data) * 0.8)
                    x_train, x_val = x_data[:val_split_idx], x_data[val_split_idx:]
                    y_train, y_val = y_data[:val_split_idx], y_data[split_idx:]

                    # Enhanced callbacks for better training
                    callbacks = [
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=15,
                            restore_best_weights=True,
                            min_delta=0.0001
                        ),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.5,
                            patience=7,
                            min_lr=0.0001,
                            cooldown=3
                        ),
                        tf.keras.callbacks.ModelCheckpoint(
                            filepath=f"temp_best_model_{symbol}.weights.h5",
                            monitor='val_loss',
                            save_weights_only=True,
                            save_best_only=True,
                            verbose=0
                        ),
                        tf.keras.callbacks.TerminateOnNaN()
                    ]

                    # Use fixed batch size and epochs for reproducibility
                    try:
                        model.fit(
                            x_train, y_train,
                            validation_data=(x_val, y_val),
                            epochs=100,
                            batch_size=32,
                            callbacks=callbacks,
                            verbose=0,
                            shuffle=True
                        )

                        # Load best weights
                        try:
                            model.load_weights(f"temp_best_model_{symbol}.weights.h5")
                            logger.info("Loaded best weights for %s", symbol)
                            # Clean up temporary file
                            os.remove(f"temp_best_model_{symbol}.weights.h5")
                        except Exception as e:
                            logger.warning("Could not load best weights: %s", e)

                    except Exception as training_error:
                        logger.error("Error training model for %s: %s", symbol, training_error)

                        # Try a simpler model if the main one fails
                        logger.info("Attempting to train fallback model...")
                        try:
                            # Simple fallback model
                            simple_model = Sequential()
                            simple_model.add(LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2])))
                            simple_model.add(Dense(16, activation='relu'))
                            simple_model.add(Dense(1))
                            simple_model.compile(optimizer='adam', loss='mse')

                            # Train with minimal settings
                            simple_history = simple_model.fit(
                                x_train_final, y_train_final,
                                validation_data=(x_val, y_val),
                                epochs=50,
                                batch_size=32,
                                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                                verbose=0
                            )

                            model = simple_model
                            logger.info("Successfully trained fallback model")
                        except Exception as fallback_error:
                            logger.error("Fallback model also failed: %s", fallback_error)
                            return self.predict_without_ml(data, fundamentals)

                    # Save model data
                    model_data = {
                        'last_price': data['Close'].iloc[-1],
                        'last_updated': datetime.now(),
                        'input_shape': input_shape,  # Store input shape for compatibility check
                        'sequence_length': sequence_length  # Store sequence length used for training
                    }

                    # Cache and save the model
                    self.models[symbol] = {'model': model, 'data': model_data}
                    self._save_model(symbol, model, model_data)
                else:
                    # Cache loaded model
                    self.models[symbol] = {'model': model, 'data': model_data}

            # Use the same sequence length that was used for training
            sequence_length = model_data.get('sequence_length', 30 if FAST_EVAL else 60)
            logger.info("Using sequence length of %s for prediction", sequence_length)

            # Apply the same feature engineering as during training
            if FAST_EVAL:
                # Process features the same way as in _prepare_lstm_data fast mode
                df = data.copy()

                # Calculate log returns (more stable than raw prices)
                df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

                # Add lagged returns (important predictors)
                for i in range(1, 6):  # Add 5 days of lagged returns
                    df[f'log_return_lag_{i}'] = df['log_return'].shift(i)

                # Calculate simple technical indicators
                df['MA5'] = df['Close'].rolling(window=5).mean()
                df['MA20'] = df['Close'].rolling(window=20).mean()
                df['MA50'] = df['Close'].rolling(window=50).mean()

                # Add moving average crossovers (boolean converted to int)
                df['MA5_cross_MA20'] = df['MA5'] > df['MA20'].astype(int)

                # Add RSI - Relative Strength Index
                df['RSI'] = df.ta.rsi(length=14)
                df['RSI_7'] = df.ta.rsi(length=7)

                # Add momentum
                df['momentum_5'] = df['Close'].pct_change(periods=5)
                df['momentum_10'] = df['Close'].pct_change(periods=10)

                # Add volatility features
                df['volatility'] = df['Close'].rolling(window=10).std() / df['Close'] * 100
                df['volatility_change'] = df['volatility'].pct_change(5)

                # Add day of week (cyclic encoding for time features)
                if isinstance(df.index, pd.DatetimeIndex):
                    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
                    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

                # Add volume features
                if 'Volume' in df.columns:
                    df['volume_change'] = df['Volume'].pct_change()
                    df['volume_ma5'] = df['Volume'].rolling(window=5).mean()
                    df['rel_volume'] = df['Volume'] / df['volume_ma5']

                # Fill NaN values
                df = df.ffill().bfill().fillna(0)

                # Keep only features for the model, dropping price and volume columns
                drop_cols = ['Open', 'High', 'Low', 'Close',
    'Volume'] if 'Volume' in df.columns else ['Open', 'High', 'Low', 'Close']
                features_df = df.drop(drop_cols, axis=1, errors='ignore')
            else:
                # Use the full feature set similar to _prepare_lstm_data's full mode
                df = data.copy()
                # ... (similar code for the full feature set)
                # Just handle the minimal requirements here since this part can get complex

                # Fill missing values
                df = df.ffill().bfill()

                # Basic features needed
                df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

                # Add key technical indicators that should be in feature_names
                if not self.feature_names:
                    logger.warning("No feature names available. Using basic feature set.")
                    # Basic feature set
                    df['RSI_14'] = df.ta.rsi(length=14)
                    df['MA20'] = df['Close'].rolling(window=20).mean()
                    df['volatility'] = df['log_return'].rolling(window=20).std()

                    drop_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    features_df = df.drop(drop_cols, axis=1, errors='ignore')
                else:
                    # Try to recreate all features from feature_names
                    calculated_features = {}

                    # Calculate common features that might be in feature_names
                    # RSI indicators
                    rsi_lengths = [7, 14, 21]
                    for length in rsi_lengths:
                        feature_name = f'RSI_{length}'
                        if feature_name in self.feature_names:
                            df[feature_name] = df.ta.rsi(length=length)
                            calculated_features[feature_name] = True

                    # Moving averages
                    ma_lengths = [5, 10, 20, 50, 100, 200]
                    for length in ma_lengths:
                        feature_name = f'MA{length}'
                        if feature_name in self.feature_names:
                            df[feature_name] = df['Close'].rolling(window=length).mean()
                            calculated_features[feature_name] = True

                    # Moving average crossovers
                    crossover_pairs = [(5, 20), (10, 50), (20, 50), (50, 200)]
                    for short, long in crossover_pairs:
                        feature_name = f'MA{short}_cross_MA{long}'
                        if feature_name in self.feature_names:
                            df[feature_name] = (df['Close'].rolling(window=short).mean() >
                                             df['Close'].rolling(window=long).mean()).astype(int)
                            calculated_features[feature_name] = True

                    # Volatility features
                    if 'volatility_regime' in self.feature_names:
                        df['volatility_regime'] = df['log_return'].rolling(window=20).std()
                        calculated_features['volatility_regime'] = True

                    # Time features
                    if isinstance(df.index, pd.DatetimeIndex):
                        time_features = ['day_sin', 'day_cos', 'month_sin', 'month_cos']
                        for feature in time_features:
                            if feature in self.feature_names:
                                if feature == 'day_sin':
                                    df[feature] = np.sin(2 * np.pi * df.index.dayofweek / 7)
                                elif feature == 'day_cos':
                                    df[feature] = np.cos(2 * np.pi * df.index.dayofweek / 7)
                                elif feature == 'month_sin':
                                    df[feature] = np.sin(2 * np.pi * df.index.month / 12)
                                elif feature == 'month_cos':
                                    df[feature] = np.cos(2 * np.pi * df.index.month / 12)
                                calculated_features[feature] = True

                    # Prepare the final feature DataFrame
                    features_df = pd.DataFrame()

                    # Add all the required features from feature_names
                    for feature in self.feature_names:
                        if feature in df.columns:
                            features_df[feature] = df[feature]
                        elif feature not in calculated_features:
                            # For features we couldn't calculate, add zeros
                            features_df[feature] = 0
                            logger.warning("Could not calculate feature %s, using zeros", feature)

            # Scale features using the same scaler as during training
            try:
                scaled_features = self.feature_scaler.transform(features_df)
            except ValueError as scale_error:
                # If there's a shape mismatch, log detailed information and fallback
                logger.error("Feature shape mismatch for %s: expected %s features, got %s", symbol, self.feature_scaler.n_features_in_, features_df.shape[1])
                logger.error("Available columns: %s", features_df.columns.tolist())
                logger.error("Expected features: %s", self.feature_names)
                return self.predict_without_ml(data, fundamentals)

            # Create input sequence for prediction (using saved sequence_length)
            if len(scaled_features) < sequence_length:
                logger.warning("Not enough data points for sequence prediction for %s", symbol)
                return self.predict_without_ml(data, fundamentals)

            x_test = scaled_features[-sequence_length:].reshape(1, sequence_length, -1)

            # Check if input shape matches model's expected input shape
            expected_shape = model_data.get('input_shape', None)
            if expected_shape and (x_test.shape[1:] != expected_shape):
                logger.error(f"Input shape mismatch for {symbol}: expected {expected_shape}, got {x_test.shape[1:]}")
                return self.predict_without_ml(data, fundamentals)

            # Make prediction with multiple runs for stability
            predictions = []
            try:
                for _ in range(5):  # Run prediction 5 times
                    predicted_scaled = model.predict(x_test, verbose=0)[0][0]
                    predictions.append(predicted_scaled)

                # Average the predictions for stability
                predicted_scaled_avg = sum(predictions) / len(predictions)

                # Transform back to original scale using the close_scaler (for log returns)
                predicted_log_return = self.close_scaler.inverse_transform([[predicted_scaled_avg]])[0][0]

                # Convert log return to price
                current_price = data['Close'].iloc[-1]
                predicted_price = current_price * np.exp(predicted_log_return)

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
            except Exception as prediction_error:
                logger.error("Error making prediction for %s: %s", symbol, prediction_error)
                # Fallback to non-ML prediction
                return self.predict_without_ml(data, fundamentals)

        except Exception as e:
            logger.error("Error in LSTM prediction for %s: %s", symbol, str(e))
            # Log traceback for detailed debugging
            import traceback
            logger.error("Traceback: %s", traceback.format_exc())
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
                logger.warning("Not enough data for %s", symbol)
                return {
                    "success": False,
                    "error": "Not enough historical data available"
                }

            # Get fundamentals with caching
            fundamentals = self._get_fundamentals(symbol)

            # Use ML-based prediction
            logger.info("Generating LSTM prediction for %s", symbol)
            try:
                prediction = self.predict_with_lstm(symbol, data, fundamentals)
            except Exception as e:
                logger.error("LSTM prediction failed for %s: %s", symbol, e)
                # Fallback to technical analysis
                prediction = self.predict_without_ml(data, fundamentals)

            # Format prediction values for consistency
            prediction["current_price"] = round(self._safe_float_conversion(prediction["current_price"]), 2)
            prediction["predicted_price"] = round(self._safe_float_conversion(prediction["predicted_price"]), 2)
            prediction["prediction_score"] = round(self._safe_float_conversion(prediction["prediction_score"]), 1)

            return prediction

        except Exception as e:
            logger.error("Error predicting for %s: %s", symbol, str(e))
            return {
                "success": False,
                "error": f"Failed to generate prediction: {str(e)}"
            }

    def evaluate_model(self, symbol, test_size=0.2):
        """Evaluate model accuracy and check for overfitting with improved stability"""
        logger.info("Evaluating model for %s...", symbol)

        try:
            # Fetch more data for better training and evaluation
            # In fast mode, use less data
            period = "1y" if FAST_EVAL else "2y"  # Use at least 1 year of data for better training
            data = self._fetch_stock_data(symbol, period=period)

            if data is None or data.empty or len(data) < 100:
                logger.warning("Not enough data for evaluation of %s", symbol)
                return {
                    "success": False,
                    "error": "Not enough historical data for proper evaluation"
                }
            
            # Check for NaN or infinite values in data
            if data.isnull().any().any() or np.isinf(data.values).any():
                logger.warning("Data contains NaN or infinite values - attempting to clean")
                data = data.fillna(method='ffill').fillna(method='bfill')
                data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Prepare data with enhanced features
            # In fast mode, use a shorter sequence length
            seq_length = 30 if FAST_EVAL else 60
            x_data, y_data, close_prices, scalers = self._prepare_lstm_data(data,
    sequence_length=seq_length, return_scalers=True)

            if len(x_data) < seq_length * 2:
                logger.warning("Not enough processed data sequences for %s", symbol)
                return {
                    "success": False,
                    "error": "Not enough processed data after feature engineering"
                }

            # Split data into training and testing sets
            split_idx = int(len(x_data) * (1 - test_size))
            x_train, x_test = x_data[:split_idx], x_data[split_idx:]
            y_train, y_test = y_data[:split_idx], y_data[split_idx:]

            # Further split training data for validation
            val_split_idx = int(len(x_train) * 0.8)
            x_train_final, x_val = x_train[:val_split_idx], x_train[val_split_idx:]
            y_train_final, y_val = y_train[:val_split_idx], y_train[val_split_idx:]

            # Run multiple models with different random seeds and average predictions (ensemble approach)
            # Increase ensemble size for better price prediction
            num_models = 2 if FAST_EVAL else 3  # Reduced for faster evaluation
            models = []
            predictions = []

            # Train models individually without cross-validation to simplify
            for i in range(num_models):
                # Set different seeds for each model
                seed = 42 + i * 13
                tf.random.set_seed(seed)
                np.random.seed(seed)

                # Build model with consistent configuration
                model = self._build_lstm_model((x_train.shape[1], x_train.shape[2]))

                # Enhanced callbacks for training
                callbacks = [
                    # Early stopping with restored weights
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=20,
                        restore_best_weights=True,
                        min_delta=0.0001
                    ),

                    # Learning rate scheduler
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.7,
                        patience=10,
                        min_lr=0.00005,
                        cooldown=3,
                        verbose=0
                    ),

                    # Terminate on NaN
                    tf.keras.callbacks.TerminateOnNaN()
                ]

                # Train the model
                try:
                    history = model.fit(
                        x_train_final, y_train_final,
                        validation_data=(x_val, y_val),
                        epochs=100,
                        batch_size=16,
                        callbacks=callbacks,
                        verbose=0,
                        shuffle=True
                    )

                    # Evaluate on train and test
                    train_loss = model.evaluate(x_train, y_train, verbose=0)
                    test_loss = model.evaluate(x_test, y_test, verbose=0)

                    # Log metrics
                    train_rmse = np.sqrt(train_loss)
                    test_rmse = np.sqrt(test_loss)
                    logger.info(f"Model {i+1} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

                    # Store model
                    models.append({
                        'model': model,
                        'history': history,
                        'train_loss': train_loss,
                        'test_loss': test_loss
                    })

                    # Make predictions
                    predictions.append(model.predict(x_test, verbose=0).flatten())

                except Exception as training_error:
                    logger.error("Error training model %s: %s", i+1, training_error)

                    if i == 0:
                        # Try a simpler model if the first one fails
                        logger.info("Trying a simpler fallback model...")
                        try:
                            # Simple fallback model
                            simple_model = Sequential()
                            simple_model.add(LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2])))
                            simple_model.add(Dense(16, activation='relu'))
                            simple_model.add(Dense(1))
                            simple_model.compile(optimizer='adam', loss='mse')

                            # Train with minimal settings
                            simple_history = simple_model.fit(
                                x_train_final, y_train_final,
                                validation_data=(x_val, y_val),
                                epochs=50,
                                batch_size=32,
                                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                                verbose=0
                            )

                            # Evaluate
                            train_loss = simple_model.evaluate(x_train, y_train, verbose=0)
                            test_loss = simple_model.evaluate(x_test, y_test, verbose=0)

                            # Store model
                            models.append({
                                'model': simple_model,
                                'history': simple_history,
                                'train_loss': train_loss,
                                'test_loss': test_loss
                            })

                            # Make predictions
                            predictions.append(simple_model.predict(x_test, verbose=0).flatten())

                            logger.info("Successfully trained fallback model")
                        except Exception as fallback_error:
                            logger.error("Fallback model also failed: %s", fallback_error)

                    continue

            # If all models failed, return error
            if not models:
                logger.error("All models failed to train properly")
                return {
                    "success": False,
                    "error": "Failed to train models properly"
                }

            # Ensemble prediction with weighted average based on test loss
            # Lower test loss = higher weight
            if len(models) > 1:
                # Calculate weights inversely proportional to test loss
                test_losses = np.array([m['test_loss'] for m in models])
                weights = 1.0 / (test_losses + 1e-10)  # Add small epsilon to avoid division by zero
                weights = weights / np.sum(weights)  # Normalize weights to sum to 1

                # Apply weighted average for ensemble prediction
                y_pred_ensemble = np.zeros_like(predictions[0])
                for i, pred in enumerate(predictions):
                    y_pred_ensemble += weights[i] * pred

                logger.info("Using weighted ensemble of %s models with weights: %s", len(models), weights)
            else:
                # Simple case: only one model
                y_pred_ensemble = predictions[0]

            # Calculate directional accuracy based on log returns
            actual_direction = np.sign(y_test)
            pred_direction = np.sign(y_pred_ensemble)

            # Handle any potential NaN values in the directions
            if np.isnan(actual_direction).any() or np.isnan(pred_direction).any():
                logger.warning("NaN values found in direction calculation, cleaning them up")
                actual_direction = np.nan_to_num(actual_direction)
                pred_direction = np.nan_to_num(pred_direction)

            directional_accuracy = np.mean(actual_direction == pred_direction) * 100

            # Calculate metrics for the ensemble
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            # Extract the close_scaler from the returned scalers
            close_scaler = scalers['close_scaler'] if 'close_scaler' in scalers else self.close_scaler

            # Get the actual test period prices directly from the dataset
            actual_prices = data['Close'].iloc[split_idx:split_idx+len(y_test)].values

            # Convert log returns back to price predictions using a simpler approach
            try:
                # We'll predict price values based on the actual stock prices
                y_test_prices = actual_prices

                # For predictions, get the last training price
                last_price_train = close_prices[split_idx-1]

                # Calculate price predictions using a naive approach - just forecast tomorrow's price
                # from today's actual price and our predicted return
                y_pred_prices = []
                for i, scaled_return in enumerate(y_pred_ensemble):
                    # Get the previous actual price as base
                    if i == 0:
                        prev_price = last_price_train
                    else:
                        prev_price = actual_prices[i-1]

                    # Unscale the prediction (if needed)
                    try:
                        unscaled_return = close_scaler.inverse_transform([[scaled_return]])[0][0]
                    except:
                        unscaled_return = scaled_return

                    # Calculate the next price
                    next_price = prev_price * (1 + unscaled_return)
                    y_pred_prices.append(next_price)

                # Ensure arrays are same length
                min_len = min(len(y_test_prices), len(y_pred_prices))
                y_test_prices = y_test_prices[:min_len]
                y_pred_prices = y_pred_prices[:min_len]

                # Log some prices for debugging
                logger.info(f"Actual prices: First={y_test_prices[0]:.2f}, Last={y_test_prices[-1]:.2f}")
                logger.info(f"Predicted prices: First={y_pred_prices[0]:.2f}, Last={y_pred_prices[-1]:.2f}")

            except Exception as e:
                logger.error("Error in price reconstruction: %s. Using naive approach.", e)

                # Fallback to naive predictions
                y_test_prices = actual_prices
                y_pred_prices = np.ones_like(actual_prices) * actual_prices.mean()

            # Calculate daily price changes for directional accuracy
            y_test_changes = np.diff(y_test_prices)
            y_pred_changes = np.diff(y_pred_prices)

            # Calculate directional accuracy based on actual daily changes
            actual_direction = np.sign(y_test_changes)
            pred_direction = np.sign(y_pred_changes)

            # Handle any NaN values
            actual_direction = np.nan_to_num(actual_direction)
            pred_direction = np.nan_to_num(pred_direction)

            # Directional accuracy is how often we correctly predict up/down movement
            matches = (actual_direction == pred_direction)
            directional_accuracy = np.mean(matches) * 100

            # Calculate a naive baseline (always predict same as yesterday)
            naive_direction = np.sign(np.diff(actual_prices[:-1]))
            naive_accuracy = np.mean(naive_direction == actual_direction[:-1]) * 100 if len(naive_direction) > 0 else 50.0

            # Calculate metrics based on prices
            rmse = np.sqrt(mean_squared_error(y_test_prices, y_pred_prices))
            mae = mean_absolute_error(y_test_prices, y_pred_prices)

            # MAPE calculation with protection against zeros
            epsilon = 1e-10
            mape = np.mean(np.abs((y_test_prices - y_pred_prices) / (y_test_prices + epsilon))) * 100

            # Calculate normalized RMSE
            mean_price = np.mean(y_test_prices)
            normalized_rmse = rmse / mean_price * 100 if mean_price > 0 else 100.0

            # Calculate baselines for price prediction
            # Naive prediction (today = yesterday)
            naive_pred = np.roll(y_test_prices, 1)
            naive_pred[0] = y_test_prices[0]  # Set first entry
            naive_rmse = np.sqrt(mean_squared_error(y_test_prices, naive_pred))

            # Mean prediction (always predict mean price)
            mean_pred = np.ones_like(y_test_prices) * np.mean(y_test_prices)
            mean_rmse = np.sqrt(mean_squared_error(y_test_prices, mean_pred))

            # Calculate R² relative to naive prediction instead of mean
            # This is more appropriate for time series
            ss_total_naive = np.sum((y_test_prices - naive_pred) ** 2)
            ss_residual = np.sum((y_test_prices - y_pred_prices) ** 2)

            if ss_total_naive > epsilon:
                r2_price_vs_naive = 1 - (ss_residual / ss_total_naive)
            else:
                r2_price_vs_naive = 0.0

            # Also calculate traditional R² (vs mean)
            ss_total_mean = np.sum((y_test_prices - mean_pred) ** 2)
            if ss_total_mean > epsilon:
                r2_price = 1 - (ss_residual / ss_total_mean)
            else:
                r2_price = 0.0

            # For returns-based R², calculate daily returns
            actual_returns = y_test_changes / y_test_prices[:-1]
            pred_returns = y_pred_changes / y_pred_prices[:-1]

            if len(actual_returns) > 0:
                # Calculate R² for returns
                try:
                    r2_returns = r2_score(actual_returns, pred_returns)
                except:
                    r2_returns = 0.0
            else:
                r2_returns = 0.0

            # Also calculate improvement over naive
            improvement_over_naive = ((naive_rmse - rmse) / naive_rmse) * 100 if naive_rmse > 0 else 0

            # Calculate classification metrics with error handling
            try:
                # Create binary classifications for up/down movements
                y_true_bin = (np.sign(actual_direction) > 0).astype(int)
                y_pred_bin = (np.sign(pred_direction) > 0).astype(int)
                
                # Replace any NaN values
                y_true_bin = np.nan_to_num(y_true_bin)
                y_pred_bin = np.nan_to_num(y_pred_bin)
                
                accuracy = accuracy_score(y_true_bin, y_pred_bin)
                precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
                recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
                f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            except Exception as e:
                logger.warning("Error calculating classification metrics: %s. Using default values.", e)
                accuracy = directional_accuracy / 100  # Fall back to directional accuracy
                precision = 0.5
                recall = 0.5
                f1 = 0.5

            # Calculate average model metrics
            avg_train_loss = np.mean([m['train_loss'] for m in models])
            avg_test_loss = np.mean([m['test_loss'] for m in models])

            # Check for overfitting (using less strict threshold due to ensemble)
            is_overfitting = avg_train_loss < avg_test_loss * 0.7

            # Check for underfitting based on R² and directional accuracy
            is_underfitting = r2_price < 0.1 or directional_accuracy < 55

            # Print comprehensive results
            print("\n" + "="*60)
            print(f"ENSEMBLE MODEL EVALUATION RESULTS FOR {symbol}")
            print("="*60)
            print(f"Number of models in ensemble: {len(models)}")
            print(f"Avg Training Loss: {avg_train_loss:.6f}")
            print(f"Avg Testing Loss: {avg_test_loss:.6f}")
            print(f"RMSE: ${rmse:.2f}")
            print(f"MAE: ${mae:.2f}")
            print(f"MAPE: {mape:.2f}%")
            print(f"R² Score (prices): {r2_price:.4f}")
            print(f"R² Score (vs naive): {r2_price_vs_naive:.4f}")
            print(f"R² Score (returns): {r2_returns:.4f}")
            print(f"Normalized RMSE: {normalized_rmse:.2f}%")
            print(f"Improvement over naive: {improvement_over_naive:.2f}%")
            print(f"Directional Accuracy: {directional_accuracy:.2f}%")
            print(f"Naive Directional Accuracy: {naive_accuracy:.2f}%")
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
                print("MODEL STATUS: GOOD FIT - Model ensemble appears to be properly trained")

        except Exception as e:
            logger.error("Error in model evaluation for %s: %s", symbol, str(e))
            return {
                "success": False,
                "error": f"Failed to evaluate model: {str(e)}"
            }

        return {
            "success": True,
            "directional_accuracy": directional_accuracy,
            "r2_returns": r2_returns,
            "r2": r2_price,
            "normalized_rmse": normalized_rmse,
            "is_overfitting": is_overfitting,
            "is_underfitting": is_underfitting
        }

    def simplified_evaluate(self, symbol):
        """A simplified evaluation that's more robust against data issues"""
        logger.info("Running simplified evaluation for %s...", symbol)
        
        try:
            # Get data
            data = self._fetch_stock_data(symbol, period="1y")
            
            if data is None or data.empty or len(data) < 60:
                return {
                    "success": False,
                    "error": "Not enough historical data"
                }
            
            # Clean data
            data = data.fillna(method='ffill').fillna(method='bfill')
            data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Calculate basic metrics
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility in %
            
            # Calculate simple moving averages
            ma50 = data['Close'].rolling(50).mean().iloc[-1]
            ma200 = data['Close'].rolling(200).mean().iloc[-1]
            
            # Market trend
            trend = "Bullish" if ma50 > ma200 else "Bearish"
            
            # Last month performance
            last_month = data['Close'].iloc[-21:].pct_change(20).iloc[-1] * 100
            
            # Simple metrics
            current_price = data['Close'].iloc[-1]
            avg_volume = data['Volume'].mean()
            
            # Print results
            print("\n" + "="*60)
            print(f"SIMPLIFIED EVALUATION FOR {symbol}")
            print("="*60)
            print(f"Current Price: ${current_price:.2f}")
            print(f"Trend (MA50 vs MA200): {trend}")
            print(f"Last Month Performance: {last_month:.2f}%")
            print(f"Annualized Volatility: {volatility:.2f}%")
            print(f"Average Daily Volume: {avg_volume:.0f}")
            print("-"*60)
            
            # Return success
            return {
                "success": True,
                "price": current_price,
                "trend": trend,
                "volatility": volatility
            }
            
        except Exception as e:
            logger.error("Error in simplified evaluation: %s", str(e))
            return {
                "success": False,
                "error": f"Evaluation failed: {str(e)}"
            }

    def simple_backtest(self, symbol, test_days=30):
        """Simple and robust backtesting to evaluate model performance"""
        logger.info(f"Running simple backtest for {symbol} over last {test_days} days")
        
        try:
            # Fetch data with longer history
            data = self._fetch_stock_data(symbol, period="1y")
            
            if data is None or data.empty or len(data) < test_days + 30:
                return {
                    "success": False,
                    "error": "Not enough data for backtesting"
                }
            
            # Clean data
            data = data.ffill().bfill()
            data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Split into training and testing
            train_data = data.iloc[:-test_days].copy()
            test_data = data.iloc[-test_days:].copy()
            
            # Train a model on training data and predict on test days
            predictions = []
            actual_prices = test_data['Close'].values
            dates = test_data.index
            
            # Initialize a simple model - a straightforward, small LSTM for stability
            try:
                # Prepare training data
                lookback = 10  # Use shorter lookback for stability
                X = []
                y = []
                
                # Simple features - just returns, moving averages, and volume
                train_features = train_data.copy()
                train_features['returns'] = train_features['Close'].pct_change()
                train_features['ma5'] = train_features['Close'].rolling(5).mean()
                train_features['ma20'] = train_features['Close'].rolling(20).mean()
                
                # Create sequences for LSTM
                for i in range(lookback, len(train_features)):
                    # Simple feature set for stability
                    feature_set = [
                        train_features['returns'].iloc[i-lookback:i].values,
                        train_features['Close'].iloc[i-lookback:i].values / train_features['Close'].iloc[i-lookback],
                        train_features['Volume'].iloc[i-lookback:i].values / train_features['Volume'].iloc[i-lookback] if 'Volume' in train_features.columns else np.ones(lookback)
                    ]
                    X.append(np.column_stack(feature_set))
                    y.append(train_features['returns'].iloc[i])
                
                X = np.array(X)
                y = np.array(y)
                
                # Simple LSTM model
                model = Sequential([
                    LSTM(20, input_shape=(X.shape[1], X.shape[2])),
                    Dense(10, activation='relu'),
                    Dense(1)
                ])
                
                model.compile(optimizer='adam', loss='mse')
                
                # Train with early stopping
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='loss', patience=10, restore_best_weights=True
                )
                
                model.fit(
                    X, y, 
                    epochs=50, 
                    batch_size=32, 
                    callbacks=[early_stop], 
                    verbose=0
                )
                
                # Make predictions on test data
                predicted_returns = []
                
                # Prepare test data same way as training
                test_features = test_data.copy()
                test_features['returns'] = test_features['Close'].pct_change()
                test_features['ma5'] = test_features['Close'].rolling(5).mean()
                test_features['ma20'] = test_features['Close'].rolling(20).mean()
                
                # Initial window is from end of training data
                window = train_features.iloc[-lookback:].copy()
                
                # Predict each day in test period
                for i in range(len(test_data)):
                    if i > 0:
                        # Update window with actual test data
                        window = window.iloc[1:].copy()
                        window = pd.concat([window, test_features.iloc[i-1:i]])
                    
                    # Create feature set for prediction
                    feature_set = [
                        window['returns'].values,
                        window['Close'].values / window['Close'].iloc[0],
                        window['Volume'].values / window['Volume'].iloc[0] if 'Volume' in window.columns else np.ones(lookback)
                    ]
                    
                    X_pred = np.array([np.column_stack(feature_set)])
                    
                    # Predict return
                    predicted_return = model.predict(X_pred, verbose=0)[0][0]
                    predicted_returns.append(predicted_return)
                
                # Convert returns to prices
                predicted_prices = []
                last_price = train_data['Close'].iloc[-1]
                
                for ret in predicted_returns:
                    next_price = last_price * (1 + ret)
                    predicted_prices.append(next_price)
                    last_price = next_price
                
                # Calculate metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                # Basic price metrics
                mse = mean_squared_error(actual_prices, predicted_prices)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(actual_prices, predicted_prices)
                
                # Normalize by price level
                mean_price = np.mean(actual_prices)
                normalized_rmse = (rmse / mean_price) * 100 if mean_price > 0 else 100
                
                # R² score
                r2 = r2_score(actual_prices, predicted_prices)
                
                # Directional accuracy
                actual_direction = np.sign(np.diff(np.append([train_data['Close'].iloc[-1]], actual_prices)))
                pred_direction = np.sign(np.diff(np.append([train_data['Close'].iloc[-1]], predicted_prices)))
                
                # Clean up any NaN values
                actual_direction = np.nan_to_num(actual_direction)
                pred_direction = np.nan_to_num(pred_direction)
                
                directional_accuracy = np.mean(actual_direction == pred_direction) * 100
                
                # Print results
                print("\n" + "="*60)
                print(f"BACKTEST RESULTS FOR {symbol} - Last {test_days} Days")
                print("="*60)
                print(f"Final Price: ${actual_prices[-1]:.2f}")
                print(f"Predicted Final Price: ${predicted_prices[-1]:.2f}")
                print(f"RMSE: ${rmse:.2f}")
                print(f"MAE: ${mae:.2f}")
                print(f"Normalized RMSE: {normalized_rmse:.2f}%")
                print(f"R² Score: {r2:.4f}")
                print(f"Directional Accuracy: {directional_accuracy:.2f}%")
                print("-"*60)
                
                # Return results
                return {
                    "success": True,
                    "symbol": symbol,
                    "rmse": rmse,
                    "normalized_rmse": normalized_rmse,
                    "r2": r2,
                    "directional_accuracy": directional_accuracy
                }
                
            except Exception as model_error:
                logger.error(f"Error in model training/prediction: {model_error}")
                # Fall back to a naive prediction
                print("\n" + "="*60)
                print(f"SIMPLIFIED METRICS FOR {symbol} (Fallback - Model Failed)")
                print("="*60)
                
                # Calculate some basic stats on the data
                returns = data['Close'].pct_change().dropna()
                daily_volatility = returns.std()
                annualized_volatility = daily_volatility * np.sqrt(252) * 100
                
                print(f"Current Price: ${data['Close'].iloc[-1]:.2f}")
                print(f"Daily Volatility: {daily_volatility:.4f}")
                print(f"Annualized Volatility: {annualized_volatility:.2f}%")
                print(f"Last Month Return: {(data['Close'].iloc[-1] / data['Close'].iloc[-22] - 1) * 100:.2f}%")
                
                return {
                    "success": False,
                    "error": f"Model training failed: {str(model_error)}",
                    "current_price": data['Close'].iloc[-1]
                }
                
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return {
                "success": False,
                "error": f"Backtesting failed: {str(e)}"
            }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stock price prediction using machine learning")
    parser.add_argument("symbol", help="Stock symbol (e.g., MSFT, AAPL)")
    parser.add_argument("--metrics", action="store_true", help="Show basic metrics and evaluation")
    parser.add_argument("--full-metrics", action="store_true", help="Show detailed ML model metrics (may fail with some data)")
    parser.add_argument("--backtest", action="store_true", help="Perform backtesting to show accuracy and R2 score")
    parser.add_argument("--test-days", type=int, default=30, help="Number of days to use for backtesting")
    parser.add_argument("--period", default="6mo", help="Data period (e.g., 3mo, 6mo, 1y, 2y)")
    parser.add_argument("--fast", action="store_true", help="Use fast evaluation mode")
    
    args = parser.parse_args()
    
    if args.fast:
        FAST_EVAL = True
        print("Fast evaluation mode enabled")
    
    predictor = StockPredictor()
    
    if args.backtest:
        print(f"Running backtesting for {args.symbol} over last {args.test_days} days...")
        result = predictor.simple_backtest(args.symbol, test_days=args.test_days)
        if not result["success"]:
            print(f"Error: {result['error']}")
    elif args.full_metrics:
        print(f"Evaluating detailed metrics for {args.symbol}...")
        result = predictor.evaluate_model(args.symbol)
        if not result["success"]:
            print(f"Error: {result['error']}")
    elif args.metrics:
        print(f"Evaluating basic metrics for {args.symbol}...")
        result = predictor.simplified_evaluate(args.symbol)
        if not result["success"]:
            print(f"Error: {result['error']}")
    else:
        print(f"Generating prediction for {args.symbol}...")
        result = predictor.train_and_predict(args.symbol)
        if result["success"]:
            print(f"\nPrediction for {args.symbol}:")
            print(f"Current price: ${result['current_price']:.2f}")
            print(f"Predicted price: ${result['predicted_price']:.2f}")
            print(f"Direction: {result['direction'].upper()}")
            print(f"Confidence: {result['confidence']}")
            print(f"Score: {result['prediction_score']}/100")
            print(f"\nFactors: {result['factors']}")
        else:
            print(f"Error: {result['error']}")
