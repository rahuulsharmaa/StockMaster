from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime
from sqlalchemy.orm import Session
from database.connection import get_db
from utils.auth_utils import get_current_user
from database.models import StockSearch, User
from typing import Optional

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Helper to get current user from cookies
async def get_current_user_from_cookie(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token or not token.startswith("Bearer "):
        return None
    
    try:
        token = token.split("Bearer ")[1]
        user = await get_current_user(token, db)
        return user
    except HTTPException:
        return None

# Protected page helper
def protected_page(template_name: str):
    async def endpoint(request: Request, db: Session = Depends(get_db)):
        user = await get_current_user_from_cookie(request, db)
        if not user:
            return RedirectResponse(url="/login")
        
        return templates.TemplateResponse(
            template_name,
            {
                "request": request,
                "current_year": datetime.now().year,
                "user": user
            }
        )
    return endpoint

@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request, 
    db: Session = Depends(get_db),
    login_error: Optional[str] = None,
    signup_error: Optional[str] = None,
    registered: Optional[bool] = None
):
    user = await get_current_user_from_cookie(request, db)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "current_year": datetime.now().year,
            "user": user,
            "login_error": login_error,
            "signup_error": signup_error,
            "registered": registered
        }
    )

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    # Just redirect to home with modal parameter
    return RedirectResponse(url="/?show_login=true")


@router.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    # Just redirect to home with modal parameter
    return RedirectResponse(url="/?show_signup=true")
@router.get("/about", response_class=HTMLResponse)
async def about_page(request: Request, db: Session = Depends(get_db)):
    user = await get_current_user_from_cookie(request, db)
    return templates.TemplateResponse(
        "about.html",
        {
            "request": request,
            "current_year": datetime.now().year,
            "user": user
        }
    )
# Protected page helper
def protected_page(template_name: str):
    async def endpoint(request: Request, db: Session = Depends(get_db)):
        user = await get_current_user_from_cookie(request, db)
        if not user:
            return RedirectResponse(url="/login")
        
        # For dashboard page, we need to get recent stock searches
        if template_name == "dashboard.html":
            recent_searches = db.query(StockSearch)\
                .filter(StockSearch.user_id == user.id)\
                .order_by(StockSearch.created_at.desc())\
                .all()
                
            return templates.TemplateResponse(
                template_name,
                {
                    "request": request,
                    "current_year": datetime.now().year,
                    "user": user,
                    "recent_searches": recent_searches
                }
            )
        
        # For other protected pages
        return templates.TemplateResponse(
            template_name,
            {
                "request": request,
                "current_year": datetime.now().year,
                "user": user
            }
        )
    return endpoint

# Protected routes
router.get("/dashboard", response_class=HTMLResponse)(protected_page("dashboard.html"))
router.get("/profile", response_class=HTMLResponse)(protected_page("profile.html"))
router.get("/learning", response_class=HTMLResponse)(protected_page("learning.html"))