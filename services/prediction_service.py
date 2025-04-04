# import yfinance as yf
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import logging
# import asyncio
# from concurrent.futures import ThreadPoolExecutor

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Create a global thread pool executor
# thread_pool = ThreadPoolExecutor(max_workers=2)

# class StockPredictor:
#     def __init__(self):
#         self.scaler = MinMaxScaler(feature_range=(0, 1))
        
#     def _preprocess_data(self, data):
#         """Simplified preprocessing using only recent trend"""
#         # Extract the 'Close' prices
#         close_prices = data['Close'].values
        
#         # Calculate simple moving averages
#         short_ma = data['Close'].rolling(window=5).mean().iloc[-1]
#         long_ma = data['Close'].rolling(window=20).mean().iloc[-1]
        
#         # Calculate recent momentum
#         momentum = (data['Close'].iloc[-1] / data['Close'].iloc[-10] - 1) * 100
        
#         # Calculate volatility
#         volatility = data['Close'].pct_change().std() * 100
        
#         return {
#             'current_price': close_prices[-1],
#             'short_ma': short_ma,
#             'long_ma': long_ma,
#             'momentum': momentum,
#             'volatility': volatility
#         }
        
#     def predict_without_ml(self, data, fundamentals):
#         """Make prediction without using ML"""
#         metrics = self._preprocess_data(data)
        
#         # Determine direction based on simple indicators
#         bullish_signals = 0
#         bearish_signals = 0
        
#         # Short MA > Long MA is bullish
#         if metrics['short_ma'] > metrics['long_ma']:
#             bullish_signals += 1
#         else:
#             bearish_signals += 1
            
#         # Positive momentum is bullish
#         if metrics['momentum'] > 0:
#             bullish_signals += 1
#         else:
#             bearish_signals += 1
            
#         # Lower volatility is generally better for prediction confidence
#         confidence = "Medium"
#         if metrics['volatility'] < 1.5:
#             confidence = "High"
#         elif metrics['volatility'] > 3:
#             confidence = "Low"
            
#         # Determine direction and score
#         direction = "bullish" if bullish_signals > bearish_signals else "bearish"
        
#         # Calculate prediction score (50 = neutral, 0-100 scale)
#         base_score = 50
#         momentum_factor = min(max(metrics['momentum'] * 2, -25), 25)  # Cap at ±25
#         ma_factor = 15 if (metrics['short_ma'] > metrics['long_ma']) == (direction == "bullish") else -15
        
#         prediction_score = base_score + momentum_factor + ma_factor
#         prediction_score = max(min(prediction_score, 100), 0)  # Ensure 0-100 range
        
#         # Determine factors
#         factors = self._determine_factors(data, fundamentals, direction)
        
#         return {
#             "success": True,
#             "current_price": metrics['current_price'],
#             "predicted_price": metrics['current_price'] * (1 + (0.05 if direction == "bullish" else -0.05)),
#             "direction": direction,
#             "prediction_score": prediction_score,
#             "confidence": confidence,
#             "factors": factors
#         }
    
#     def _get_fundamentals(self, stock):
#         """Get fundamental data for the stock"""
#         try:
#             # Get basic info
#             info = stock.info
            
#             fundamentals = {
#                 "pe_ratio": info.get("trailingPE", None),
#                 "forward_pe": info.get("forwardPE", None),
#                 "peg_ratio": info.get("pegRatio", None),
#                 "dividend_yield": info.get("dividendYield", 0),
#                 "beta": info.get("beta", 0),
#                 "market_cap": info.get("marketCap", 0),
#                 "profit_margin": info.get("profitMargins", 0),
#                 "return_on_equity": info.get("returnOnEquity", 0),
#                 "debt_to_equity": info.get("debtToEquity", 0)
#             }
            
#             return fundamentals
#         except Exception as e:
#             logger.warning(f"Error fetching fundamentals: {e}")
#             return {}
    
#     def _determine_factors(self, data, fundamentals, direction):
#         """Determine key factors that influenced the prediction"""
#         factors = []
        
#         # Check for recent price momentum
#         recent_returns = data['Close'].pct_change(5).iloc[-1] * 100
#         if (direction == "bullish" and recent_returns > 1) or (direction == "bearish" and recent_returns < -1):
#             factors.append("price momentum")
        
#         # Check volatility
#         volatility = data['Close'].pct_change().std() * 100
#         if volatility > 2:
#             factors.append("high volatility")
        
#         # Check trading volume
#         if 'Volume' in data.columns:
#             avg_volume = data['Volume'].mean()
#             recent_volume = data['Volume'].iloc[-5:].mean()
#             if recent_volume > avg_volume * 1.2:
#                 factors.append("increased trading volume")
        
#         # Check fundamentals
#         if fundamentals:
#             if fundamentals.get("pe_ratio"):
#                 pe = fundamentals.get("pe_ratio")
#                 if pe and pe < 15 and direction == "bullish":
#                     factors.append("attractive valuation")
#                 elif pe and pe > 30 and direction == "bearish":
#                     factors.append("high valuation")
            
#             if fundamentals.get("dividend_yield", 0) > 0.03:
#                 factors.append("dividend yield")
                
#             if fundamentals.get("beta", 1) > 1.5:
#                 factors.append("high market sensitivity")
        
#         # If we couldn't identify specific factors
#         if not factors:
#             factors = ["technical patterns", "historical price action"]
            
#         return ", ".join(factors)
        
#     def train_and_predict(self, symbol):
#         """Get prediction for a stock symbol without using ML"""
#         try:
#             # Download data
#             logger.info(f"Fetching data for {symbol}")
#             stock = yf.Ticker(symbol)
#             data = stock.history(period="1y")
            
#             if data.empty or len(data) < 30:
#                 logger.warning(f"Not enough data for {symbol}")
#                 return {
#                     "success": False,
#                     "error": "Not enough historical data available"
#                 }
            
#             # Get fundamentals
#             fundamentals = self._get_fundamentals(stock)
            
#             # Use simplified prediction method instead of ML
#             logger.info(f"Generating prediction for {symbol}")
#             return self.predict_without_ml(data, fundamentals)
            
#         except Exception as e:
#             logger.error(f"Error predicting for {symbol}: {str(e)}")
#             return {
#                 "success": False,
#                 "error": f"Failed to generate prediction: {str(e)}"
#             }

# # Initialize the predictor
# stock_predictor = StockPredictor()

# async def get_stock_prediction(symbol):
#     """Get stock prediction for the given symbol"""
#     # Run the CPU-intensive work in a thread pool
#     loop = asyncio.get_event_loop()
#     return await loop.run_in_executor(
#         thread_pool, 
#         stock_predictor.train_and_predict, 
#         symbol
#     )
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
from tensorflow.keras.layers import Dense, LSTM, Dropout
# import os
import pickle
from datetime import datetime, timedelta
import hashlib
from functools import lru_cache

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
    
    # def _prepare_lstm_data(self, data, sequence_length=60):
    #         """Prepare data for LSTM model with enhanced features"""
    # # Create feature dataframe
    # features_df = pd.DataFrame()
    
    # Price features
    def _prepare_lstm_data(self, data, sequence_length=60):
        """Prepare data for LSTM model with enhanced features."""
        features_df = pd.DataFrame()

        # Price features
        features_df['close'] = data['Close']
        features_df['high_low_ratio'] = data['High'] / data['Low']
        features_df['close_open_ratio'] = data['Close'] / data['Open']

        # Moving averages
        features_df['ma5'] = data['Close'].rolling(window=5).mean()
        features_df['ma20'] = data['Close'].rolling(window=20).mean()

        # Volatility
        features_df['volatility'] = data['Close'].pct_change().rolling(window=5).std()

        # Momentum
        features_df['momentum'] = data['Close'].pct_change(periods=5)

        # Volume features (if available)
        if 'Volume' in data.columns:
            features_df['volume_change'] = data['Volume'].pct_change()
            features_df['price_volume_ratio'] = data['Close'] * data['Volume']

        # Fill NaN values
        features_df = features_df.ffill().bfill().fillna(0)

        # Create a separate scaler for features and store it
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = self.feature_scaler.fit_transform(features_df)

        # Extract target (next day's closing price) and scale it separately
        close_prices = data['Close'].values.reshape(-1, 1)
        self.close_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_close = self.close_scaler.fit_transform(close_prices)

        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i-sequence_length:i])
            y.append(scaled_close[i, 0])

        return np.array(X), np.array(y), scaled_close

    
    def _build_lstm_model(self, input_shape):
        """Build enhanced LSTM model with regularization"""
        # Set seeds again for model consistency
        tf.random.set_seed(42)
        np.random.seed(42)
        
        model = Sequential()
        
        # First LSTM layer with more units and regularization
        model.add(LSTM(units=64, 
                       return_sequences=True, 
                       input_shape=input_shape,
                       kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(units=64, 
                       return_sequences=False,
                       kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(Dropout(0.2))
        
        # Dense layers with regularization
        model.add(Dense(units=32, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Use a more advanced optimizer with a fixed learning rate for reproducibility
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Compile model
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
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
                    
                    # Use fixed batch size and epochs for reproducibility
                    model.fit(
                        X_data, y_data,
                        epochs=50,
                        batch_size=32, 
                        callbacks=[early_stopping],
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
            prediction["current_price"] = round(float(prediction["current_price"]), 2)
            prediction["predicted_price"] = round(float(prediction["predicted_price"]), 2)
            prediction["prediction_score"] = round(float(prediction["prediction_score"]), 1)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to generate prediction: {str(e)}"
            }

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