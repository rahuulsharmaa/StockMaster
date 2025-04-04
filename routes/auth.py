from fastapi import APIRouter, Depends, HTTPException, status, Form, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from datetime import timedelta
from database.connection import get_db
from database.crud import get_user_by_email, create_user, authenticate_user
from utils.auth_utils import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from typing import Optional

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.post("/login")
async def login(
    response: Response,
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    remember: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, email, password)
    if not user:
        return RedirectResponse(
           url="/?login_error=Incorrect+email+or+password",
           status_code=status.HTTP_303_SEE_OTHER
            # {
            #     "request": request,
            #     "error": "Incorrect email or password"
            # }
        )
    
    # Create access token
    access_token_expires = timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES * 24 if remember else ACCESS_TOKEN_EXPIRE_MINUTES
    )
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )
    
    # Set cookie with token
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        max_age=1800 if not remember else 86400,  # 30 minutes in seconds
        secure=False,  # Set to True in production with HTTPS
    )
    
    # Add cache control headers for protected pages
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response

@router.post("/signup")
async def signup(
    request: Request,
    name: str = Form(...),
    # lastName: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    terms: str = Form(...),
    db: Session = Depends(get_db)
):
     # Parse first and last name from full name
    name_parts = name.split(maxsplit=1)
    firstName = name_parts[0]
    lastName = name_parts[1] if len(name_parts) > 1 else ""
    # Validate data
    # if password != confirmPassword:
    #     return templates.TemplateResponse(
    #         "signup.html", 
    #         {
    #             "request": request,
    #             "error": "Passwords do not match"
    #         }
    #     )
    
    # Check if user already exists
    existing_user = get_user_by_email(db, email)
    if existing_user:
        return RedirectResponse(
            url="/?signup_error=Email+already+registered",
            status_code=status.HTTP_303_SEE_OTHER
            # {
            #     "request": request,
            #     "error": "Email already registered"
            # }
        )
    
    # Create new user
    user = create_user(db, firstName, lastName, email, password)
    
    # Redirect to login page
    return RedirectResponse(url="/?registered=true",  status_code=status.HTTP_303_SEE_OTHER)

@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("access_token")
    
    # Add cache control headers to prevent browser back button from showing protected pages
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response