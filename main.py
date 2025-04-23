
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime
import uvicorn

# Import database
# from routes import market_api, websocket
from database.connection import engine
from database import models

# Import routes
from routes import auth, pages,profile,market_api, websocket,stock_api,prediction_routes,analytics

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Setup static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Helper to add current year
@app.middleware("http")
async def add_current_year(request: Request, call_next):
    response = await call_next(request)
    if isinstance(response, HTMLResponse):
        response.headers["X-Current-Year"] = str(datetime.now().year)
    return response
@app.middleware("http")
async def add_cache_control_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Add cache control headers to all protected routes
    if request.url.path.startswith("/dashboard") or request.url.path.startswith("/protected"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    
    return response
# Include routers
app.include_router(auth.router)
app.include_router(profile.router)
app.include_router(pages.router)
app.include_router(websocket.router)  
app.include_router(market_api.router)  
app.include_router(stock_api.router)  
app.include_router(prediction_routes.router) 
app.include_router(analytics.router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)