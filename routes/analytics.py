from fastapi import APIRouter, HTTPException, Request, Depends,status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from typing import Optional
import logging
from datetime import datetime, timedelta

from requests import Session

from database.connection import get_db
from routes.pages import get_current_user_from_cookie
from services.stock_service import StockService
from services.prediction_service import get_stock_prediction
from database.models import StockHistory, StockPrediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Dependencies
def get_stock_service():
    return StockService()

@router.get("/analytics", response_class=HTMLResponse)
async def get_analytics_page(request: Request, db: Session = Depends(get_db)):
    """Render the analytics page."""
    user = await get_current_user_from_cookie(request, db)
    if not user:
        response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        # Add cache control headers to prevent back button access
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    
    response = templates.TemplateResponse(
        "analytics.html", 
        {
            "request": request,
            "current_year": datetime.now().year,
            "user": user
        }
    )
    
    # Add cache control headers to protect this page too
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response

@router.get("/api/stock/{symbol}")
async def get_stock_info(
    symbol: str,
    stock_service: StockService = Depends(get_stock_service)
):
    """Get basic stock information."""
    try:
        # Use search_stock method which is defined in StockService
        stock_info = stock_service.search_stock(symbol)
        if "error" in stock_info:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        # Get additional info if needed
        return {
            "name": stock_info.get("name", f"{symbol} Inc."),
            "symbol": symbol,
            "sector": stock_info.get("sector", "Technology"),
            "previousClose": stock_info.get("price", 0),
            "marketCap": stock_info.get("market_cap", 0),
            "peRatio": stock_info.get("pe_ratio", 0),
            "dividend": stock_info.get("dividend_yield", 0)
        }
    except Exception as e:
        logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/stock/{symbol}/history")
async def get_stock_history(
    symbol: str,
    period: str = "1mo",
    stock_service: StockService = Depends(get_stock_service)
):
    """Get historical stock data and prediction."""
    try:
        # Simulate historical data using yfinance
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"Historical data for {symbol} not found")
        
        # Format historical data
        dates = hist.index.strftime("%Y-%m-%d").tolist()
        prices = hist["Close"].tolist()
        
        history = {
            "dates": dates,
            "prices": prices
        }
        
        # Get prediction for the stock
        prediction_result = await get_stock_prediction(symbol)
        
        # Format response
        response = {
            "dates": history["dates"],
            "prices": history["prices"],
            "prediction": None
        }
        
        # Add prediction data if successful
        if prediction_result and prediction_result.get("success", False):
            current_price = prediction_result.get("current_price", history["prices"][-1] if history["prices"] else 0)
            predicted_price = prediction_result.get("predicted_price", current_price)
            
            if history["dates"] and len(history["dates"]) > 0:
                last_date = datetime.strptime(history["dates"][-1], "%Y-%m-%d") if isinstance(history["dates"][-1], str) else history["dates"][-1]
                future_dates = []
                future_prices = []
                
                # Add prediction points (10 days into future)
                for i in range(1, 11):
                    future_date = last_date + timedelta(days=i)
                    future_dates.append(future_date.strftime("%Y-%m-%d"))
                
                # Linear path to predicted price
                current_price = history["prices"][-1] if history["prices"] else current_price
                price_diff = predicted_price - current_price
                step = price_diff / 10
                
                for i in range(1, 11):
                    future_prices.append(current_price + (step * i))
                
                response["prediction"] = {
                    "dates": future_dates,
                    "prices": future_prices,
                    "direction": prediction_result.get("direction", "neutral"),
                    "confidence": prediction_result.get("confidence", "Low")
                }
        
        return response
    except Exception as e:
        logger.error(f"Error fetching stock history for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/predict/{symbol}")
async def predict_stock(
    symbol: str,
    stock_service: StockService = Depends(get_stock_service)
):
    """Generate prediction for a stock."""
    try:
        # Get current stock info using search_stock
        stock_info = stock_service.search_stock(symbol)
        if "error" in stock_info:
            return {"success": False, "error": f"Stock {symbol} not found"}
        
        # Get prediction using async function from prediction_service
        prediction = await get_stock_prediction(symbol)
        
        if not prediction or not prediction.get("success", False):
            return {"success": False, "error": prediction.get("error", "Failed to generate prediction")}
        
        # Get technical indicators - implementing a basic version if not available
        technical_indicators = await get_technical_indicators(symbol, stock_service)
        
        # Return the prediction with technical indicators
        return {
            "success": True,
            "current_price": prediction.get("current_price", stock_info.get("price", 0)),
            "predicted_price": prediction.get("predicted_price", 0),
            "direction": prediction.get("direction", "neutral"),
            "confidence": prediction.get("confidence", "Low"),
            "prediction_score": prediction.get("prediction_score", 50),
            "factors": prediction.get("factors", "No factors provided"),
            "technical_indicators": technical_indicators
        }
    except Exception as e:
        logger.error(f"Error predicting stock {symbol}: {str(e)}")
        return {"success": False, "error": str(e)}

@router.get("/api/similar-stocks")
async def get_similar_stocks(
    symbol: str,
    sector: Optional[str] = None,
    stock_service: StockService = Depends(get_stock_service)
):
    """Get similar stocks based on sector and performance."""
    try:
        # Implement a basic version if get_similar_stocks is not available
        similar_stocks = await get_similar_stocks_implementation(symbol, sector, stock_service)
        
        # Format the response
        stocks = []
        for stock in similar_stocks[:5]:  # Limit to 5 stocks
            stocks.append({
                "symbol": stock.get("symbol", ""),
                "name": stock.get("name", ""),
                "price": stock.get("price", 0.0),
                "change": stock.get("change_pct", 0.0)
            })
        
        return {"stocks": stocks}
    except Exception as e:
        logger.error(f"Error fetching similar stocks for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions to replace missing methods

async def get_technical_indicators(symbol: str, stock_service: StockService):
    """Generate basic technical indicators for a stock."""
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        
        # Get historical data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo")
        
        if hist.empty:
            return []
        
        # Calculate basic indicators
        indicators = []
        
        # Moving Averages
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        
        # RSI
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
        hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = hist['EMA12'] - hist['EMA26']
        hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        
        # Latest values
        latest = hist.iloc[-1]
        
        # MA Signal
        ma_signal = "Bullish" if latest['Close'] > latest['MA50'] else "Bearish"
        indicators.append({
            "name": "MA Signal",
            "value": ma_signal,
            "description": "Moving Average Signal (50-day)"
        })
        
        # RSI
        rsi_value = latest['RSI']
        rsi_status = "Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral"
        indicators.append({
            "name": "RSI",
            "value": round(rsi_value, 2),
            "status": rsi_status,
            "description": "Relative Strength Index (14-day)"
        })
        
        # MACD
        macd_signal = "Bullish" if latest['MACD'] > latest['Signal'] else "Bearish"
        indicators.append({
            "name": "MACD",
            "value": round(latest['MACD'], 2),
            "signal": round(latest['Signal'], 2),
            "status": macd_signal,
            "description": "Moving Average Convergence Divergence"
        })
        
        return indicators
    except Exception as e:
        logger.warning(f"Error calculating technical indicators: {str(e)}")
        return []

async def get_similar_stocks_implementation(symbol: str, sector: Optional[str], stock_service: StockService):
    """Get similar stocks to the given symbol."""
    try:
        # Define some sample similar stocks for popular tickers
        similar_stocks_map = {
            "AAPL": ["MSFT", "GOOGL", "FB", "AMZN"],
            "MSFT": ["AAPL", "GOOGL", "FB", "AMZN"],
            "GOOGL": ["FB", "MSFT", "AAPL", "AMZN"],
            "TSLA": ["NIO", "F", "GM", "RIVN"],
            "AMZN": ["WMT", "TGT", "EBAY", "SHOP"],
            "FB": ["SNAP", "TWTR", "PINS", "MSFT"],
        }
        
        # Get similar stock symbols
        similar_symbols = similar_stocks_map.get(symbol.upper(), ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])
        
        # Get info for each similar stock
        result = []
        for sym in similar_symbols:
            if sym != symbol.upper():
                stock_data = stock_service.search_stock(sym)
                if "error" not in stock_data:
                    result.append(stock_data)
        
        return result
    except Exception as e:
        logger.warning(f"Error finding similar stocks: {str(e)}")
        return []