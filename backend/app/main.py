from fastapi import FastAPI
from .routes import stock

app = FastAPI()

app.include_router(stock.router)
