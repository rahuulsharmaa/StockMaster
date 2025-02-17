from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..models.stock import Stock
from ..schemas import StockCreate, Stock
from ..database import SessionLocal

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/predict-stock", response_model=Stock)
def create_stock(stock: StockCreate, db: Session = Depends(get_db)):
    db_stock = Stock(symbol=stock.symbol, company_name=stock.company_name, price=stock.price)
    db.add(db_stock)
    db.commit()
    db.refresh(db_stock)
    return db_stock
