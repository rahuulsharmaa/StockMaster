from fastapi import APIRouter, Depends, Request, Form, HTTPException, status, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime
import os
import shutil
import uuid

from database.connection import get_db
from database.models import User, Profile
from utils.auth_utils import get_current_user

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "static/uploads/profile_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Get current user dependency
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
        "current_year": datetime.now().year
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
    profile_image: Optional[UploadFile] = File(None),
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
    
    # Handle profile image upload
    if profile_image and profile_image.filename:
        # Generate a unique filename to prevent conflicts
        file_extension = os.path.splitext(profile_image.filename)[1]
        unique_filename = f"user_{current_user.id}_{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(profile_image.file, buffer)
        
        # If user already has a profile image, delete the old one
        if profile.profile_image and os.path.exists(profile.profile_image):
            # Only delete if it's not a default image
            if "default" not in profile.profile_image:
                try:
                    os.remove(profile.profile_image)
                except Exception:
                    pass
        
        # Update the profile with new image path
        profile.profile_image = file_path
    
    db.commit()
    
    return RedirectResponse(url="/profile", status_code=303)

# Add a route to serve profile images
@router.get("/profile-image/{user_id}", response_class=FileResponse)
async def get_profile_image(user_id: int, db: Session = Depends(get_db)):
    profile = db.query(Profile).filter(Profile.user_id == user_id).first()
    
    if not profile or not profile.profile_image or not os.path.exists(profile.profile_image):
        # Return a default image if the user hasn't uploaded one
        return FileResponse("static/img/default-profile.png")
    
    return FileResponse(profile.profile_image)