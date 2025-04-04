from fastapi import APIRouter, Depends, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime

from database.connection import get_db
from database.models import User, Profile
from utils.auth_utils import get_current_user

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Create a dependency that will extract the token from the request
async def get_current_user_from_request(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Remove "Bearer " prefix if present
    if token.startswith("Bearer "):
        token = token[7:]
    
    return await get_current_user(token, db)

@router.get("/profile", response_class=HTMLResponse)
async def get_profile(request: Request, db: Session = Depends(get_db), current_user: User = Depends(get_current_user_from_request)):
    # Get the user's profile or create one if it doesn't exist
    profile = db.query(Profile).filter(Profile.user_id == current_user.id).first()
    
    if not profile:
        # Create a new profile for the user if one doesn't exist
        profile = Profile(user_id=current_user.id)
        db.add(profile)
        db.commit()
        db.refresh(profile)
    
    # Update the last login time
    profile.last_login = datetime.now().strftime("%B %d, %Y, %I:%M %p")
    db.commit()
    
    return templates.TemplateResponse("profile.html", {
        "request": request,
        "user": current_user,
        "profile": profile,
        "current_year": datetime.now().year  # Added to match pages.py template context
    })

@router.post("/profile/update", response_class=HTMLResponse)
async def update_profile(
    request: Request,
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: str = Form(...),
    phone: Optional[str] = Form(None),
    address: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
    state: Optional[str] = Form(None),
    zip_code: Optional[str] = Form(None),
    country: Optional[str] = Form(None),
    time_zone: Optional[str] = Form(None),
    bio: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_from_request)
):
    # Update user information
    current_user.first_name = first_name
    current_user.last_name = last_name
    current_user.email = email
    
    # Get or create profile
    profile = db.query(Profile).filter(Profile.user_id == current_user.id).first()
    if not profile:
        profile = Profile(user_id=current_user.id)
        db.add(profile)
    
    # Update profile information
    profile.phone = phone
    profile.address = address
    profile.city = city
    profile.state = state
    profile.zip_code = zip_code
    profile.country = country
    profile.time_zone = time_zone
    profile.bio = bio
    
    db.commit()
    
    return RedirectResponse(url="/profile", status_code=303)