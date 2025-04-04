from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from database.connection import get_db
from database.models import SearchActivity
from utils.auth_utils import get_current_user
from typing import Dict, Any, List

router = APIRouter()

@router.post("/search", response_model=Dict[str, Any])
async def search_stock(
    request: Request, 
    db: Session = Depends(get_db), 
    user: Any = Depends(get_current_user)
):
    form = await request.form()
    search_query = form.get("query")
    
    # Log search activity in the database
    search_log = SearchActivity(user_id=user.id, search_term=search_query)
    db.add(search_log)
    db.commit()
    
    # Perform actual search (you likely already have this logic)
    return {"message": "Search logged", "query": search_query}

@router.get("/test-activities", response_model=Dict[str, Any])
async def test_activities(
    db: Session = Depends(get_db), 
    user: Any = Depends(get_current_user)
):
    """Test endpoint to verify search activities are being logged correctly"""
    
    # Create a test search activity
    search_log = SearchActivity(user_id=user.id, search_term="TEST_ACTIVITY")
    db.add(search_log)
    db.commit()
    
    # Retrieve activities to verify
    activities = (
        db.query(SearchActivity)
        .filter(SearchActivity.user_id == user.id)
        .order_by(SearchActivity.search_date.desc())
        .limit(10)
        .all()
    )
    
    # Return activity data for debugging
    return {
        "message": "Test activity logged",
        "activities": [
            {
                "id": activity.id,
                "search_term": activity.search_term,
                "search_date": str(activity.search_date)
            } for activity in activities
        ]
    }