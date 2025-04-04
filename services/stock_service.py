# # services/stock_service.py
# import yfinance as yf
# from typing import List, Dict, Any
# import pandas as pd

# class StockService:
#     @staticmethod
#     def get_market_indices() -> List[Dict[str, Any]]:
#         """
#         Fetches current data for major market indices
#         """
#         # Define the major indices to track
#         indices = [
            
#             {"symbol": "^VIX", "name": "Volatility Index"},
#             {"symbol": "^FTSE", "name": "FTSE 100"},
#             {"symbol": "^N225", "name": "Nikkei 225"},
#             {"symbol": "^HSI", "name": "Hang Seng"},
#             {"symbol": "^GDAXI", "name": "DAX"},
#             {"symbol": "^GSPC", "name": "S&P 500"},
#             {"symbol": "^DJI", "name": "Dow Jones"},
#             {"symbol": "^IXIC", "name": "NASDAQ"},
#             {"symbol": "^RUT", "name": "Russell 2000"},
#         ]
        
#         result = []
        
#         try:
#             # Fetch data for all indices at once for efficiency
#             symbols = [index["symbol"] for index in indices]
#             data = yf.download(symbols, period="2d", group_by="ticker", progress=False)
            
#             for index in indices:
#                 symbol = index["symbol"]
#                 try:
#                     # Use iloc instead of negative indexing to avoid deprecation warning
#                     df = data[symbol]
                    
#                     if isinstance(df, pd.DataFrame) and not df.empty:
#                         # Get the last row (today) and second to last row (yesterday) if available
#                         if len(df) >= 2:
#                             current_price = round(df["Close"].iloc[-1], 2)
#                             prev_close = round(df["Close"].iloc[-2], 2)
#                         else:
#                             current_price = round(df["Close"].iloc[-1], 2)
#                             prev_close = round(df["Open"].iloc[-1], 2)
                        
#                         # Calculate percentage change
#                         change_pct = round(((current_price - prev_close) / prev_close) * 100, 2)
                        
#                         result.append({
#                             "name": index["name"],
#                             "symbol": symbol,
#                             "price": current_price,
#                             "change_pct": change_pct,
#                             "change_direction": "up" if change_pct >= 0 else "down"
#                         })
#                     else:
#                         # If dataframe is empty, add placeholder data
#                         result.append({
#                             "name": index["name"],
#                             "symbol": symbol,
#                             "price": 0,
#                             "change_pct": 0,
#                             "change_direction": "neutral",
#                             "error": "No data available"
#                         })
#                 except Exception as e:
#                     # If there's an error with a specific index, add with error info
#                     result.append({
#                         "name": index["name"],
#                         "symbol": symbol,
#                         "price": 0,
#                         "change_pct": 0,
#                         "change_direction": "neutral",
#                         "error": str(e)
#                     })
#         except Exception as e:
#             # Fallback with mock data if API call fails
#             for index in indices:
#                 result.append({
#                     "name": index["name"],
#                     "symbol": index["symbol"],
#                     "price": 0,
#                     "change_pct": 0,
#                     "change_direction": "neutral",
#                     "error": f"API fetch failed: {str(e)}"
#                 })
        
#         return result
    
    
import yfinance as yf
from typing import List, Dict, Any
import pandas as pd

class StockService:
    @staticmethod
    def get_market_indices() -> List[Dict[str, Any]]:
        """
        Fetches current data for major market indices
        """
        indices = [
            {"symbol": "^VIX", "name": "Volatility Index"},
            {"symbol": "^FTSE", "name": "FTSE 100"},
            {"symbol": "^N225", "name": "Nikkei 225"},
            {"symbol": "^HSI", "name": "Hang Seng"},
            {"symbol": "^GDAXI", "name": "DAX"},
            {"symbol": "^GSPC", "name": "S&P 500"},
            {"symbol": "^DJI", "name": "Dow Jones"},
            {"symbol": "^IXIC", "name": "NASDAQ"},
            {"symbol": "^RUT", "name": "Russell 2000"},
        ]
        
        result = []
        
        try:
            symbols = [index["symbol"] for index in indices]
            data = yf.download(symbols, period="2d", group_by="ticker", progress=False)
            
            for index in indices:
                symbol = index["symbol"]
                try:
                    df = data[symbol]
                    
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        if len(df) >= 2:
                            current_price = round(df["Close"].iloc[-1], 2)
                            prev_close = round(df["Close"].iloc[-2], 2)
                        else:
                            current_price = round(df["Close"].iloc[-1], 2)
                            prev_close = round(df["Open"].iloc[-1], 2)
                        
                        change_pct = round(((current_price - prev_close) / prev_close) * 100, 2)
                        
                        result.append({
                            "name": index["name"],
                            "symbol": symbol,
                            "price": current_price,
                            "change_pct": change_pct,
                            "change_direction": "up" if change_pct >= 0 else "down"
                        })
                    else:
                        result.append({
                            "name": index["name"],
                            "symbol": symbol,
                            "price": 0,
                            "change_pct": 0,
                            "change_direction": "neutral",
                            "error": "No data available"
                        })
                except Exception as e:
                    result.append({
                        "name": index["name"],
                        "symbol": symbol,
                        "price": 0,
                        "change_pct": 0,
                        "change_direction": "neutral",
                        "error": str(e)
                    })
        except Exception as e:
            for index in indices:
                result.append({
                    "name": index["name"],
                    "symbol": index["symbol"],
                    "price": 0,
                    "change_pct": 0,
                    "change_direction": "neutral",
                    "error": f"API fetch failed: {str(e)}"
                })
        
        return result
    
    @staticmethod
    def search_stock(symbol: str) -> Dict[str, Any]:
        """
        Searches for stock data by symbol
        """
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="2d")
            
            if hist.empty:
                return {"error": "No data available for this symbol"}
            
            if len(hist) >= 2:
                current_price = round(hist["Close"].iloc[-1], 2)
                prev_close = round(hist["Close"].iloc[-2], 2)
            else:
                current_price = round(hist["Close"].iloc[-1], 2)
                prev_close = round(hist["Open"].iloc[-1], 2)
            
            change_pct = round(((current_price - prev_close) / prev_close) * 100, 2)
            
            info = stock.info if hasattr(stock, "info") else {}
            name = info.get("shortName", info.get("longName", symbol))
            
            return {
                "name": name,
                "symbol": symbol,
                "price": current_price,
                "change_pct": change_pct,
                "change_direction": "up" if change_pct >= 0 else "down"
            }
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}