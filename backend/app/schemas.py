from pydantic import BaseModel

class StockBase(BaseModel):
    symbol: str
    company_name: str
    price: float

class StockCreate(StockBase):
    pass

class Stock(StockBase):
    id: int

    class Config:
        orm_mode = True
