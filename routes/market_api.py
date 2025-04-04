# routes/market_api.py
from fastapi import APIRouter, HTTPException
from services.market_service import get_market_data, update_market_data

router = APIRouter(prefix="/api", tags=["market"])

@router.get("/market-news")
async def market_news():
    """Get the latest market news data"""
    data = await get_market_data()
    
    # If we don't have any data yet, try to fetch it
    if not data["market_news"]:
        data = await update_market_data()
        
    return data