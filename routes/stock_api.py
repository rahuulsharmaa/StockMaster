from fastapi import APIRouter, Depends, HTTPException, Request
from services.stock_service import StockService
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from database.connection import get_db
from database.models import StockSearch
from routes.pages import get_current_user_from_cookie

router = APIRouter(
    prefix="/api/stock",
    tags=["stock"]
)

@router.get("/indices", response_model=List[Dict[str, Any]])
async def get_market_indices():
    """
    Get current market indices data
    """
    try:
        indices = StockService.get_market_indices()
        return indices
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")
    
@router.get("/search/{symbol}", response_model=Dict[str, Any])
async def search_stock(
    symbol: str, 
    request: Request, 
    db: Session = Depends(get_db)
):
    """
    Search for a specific stock by symbol
    """
    try:
        # Get current user from cookie
        user = await get_current_user_from_cookie(request, db)
        
        # Record search activity if user is logged in
        if user:
            new_search = StockSearch(
                user_id=user.id,
                symbol=symbol.upper()
            )
            db.add(new_search)
            db.commit()
        
        stock_data = StockService.search_stock(symbol)
        if "error" in stock_data:
            raise HTTPException(status_code=404, detail=stock_data["error"])
        return stock_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock data: {str(e)}")
 
@router.delete("/search/history/{search_id}")
async def delete_search_history(
    search_id: int, 
    request: Request, 
    db: Session = Depends(get_db)
):
    """
    Delete a specific search history entry
    """
    try:
        # Get current user from cookie
        user = await get_current_user_from_cookie(request, db)
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Find the search entry
        search_entry = db.query(StockSearch).filter(
            StockSearch.id == search_id,
            StockSearch.user_id == user.id
        ).first()
        
        if not search_entry:
            raise HTTPException(status_code=404, detail="Search history not found")
        
        # Delete the entry
        db.delete(search_entry)
        db.commit()
        
        return {"message": "Search history deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete search history: {str(e)}") 
    
@router.delete("/search/history", status_code=204)
async def delete_all_search_history(
    request: Request, 
    db: Session = Depends(get_db), 
    clear_all: bool = False  # Query parameter
):
    """
    Delete all search history for the current user
    """
    try:
        # Get current user from cookie
        user = await get_current_user_from_cookie(request, db)
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required")

        # Check if clear_all flag is set
        if clear_all:
            db.query(StockSearch).filter(StockSearch.user_id == user.id).delete(synchronize_session=False)
            db.commit()
            return  # No response body needed
        
        raise HTTPException(status_code=400, detail="Missing clear_all flag")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete search history: {str(e)}")
