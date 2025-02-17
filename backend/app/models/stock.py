from sqlalchemy import Column, Integer, String, Float
from ..database import Base

class Stock(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    company_name = Column(String)
    price = Column(Float)
