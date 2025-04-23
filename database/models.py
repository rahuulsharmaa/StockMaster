from datetime import datetime
from sqlalchemy import Column, Float, ForeignKey, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from .connection import Base
from sqlalchemy.orm import relationship
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
     # Relationship to Profile
    profile = relationship("Profile", back_populates="user", uselist=False)  # One-to-One Relationship

class Profile(Base):
    __tablename__ = "profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    phone = Column(String, nullable=True)
    address = Column(String, nullable=True)
    city = Column(String, nullable=True)
    state = Column(String, nullable=True)
    zip_code = Column(Integer, nullable=True)
    country = Column(String, nullable=True)
    time_zone = Column(String, nullable=True)
    bio = Column(String, nullable=True)
    subscription_status = Column(String, default="Free")  # Example: Free, Premium
    last_login = Column(String, nullable=True)
    profile_image = Column(String, nullable=True)  # Store the image path
    
    # Relationship to User
    user = relationship("User", back_populates="profile")
class StockSearch(Base):
    __tablename__ = "stock_searches"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    symbol = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship to User
    user = relationship("User", backref="stock_searches")
    
class StockHistory(Base):
    __tablename__ = "stock_history"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    date = Column(DateTime)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<StockHistory(symbol='{self.symbol}', date='{self.date}')>"

class StockPrediction(Base):
    __tablename__ = "stock_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    current_price = Column(Float)
    predicted_price = Column(Float)
    direction = Column(String)  # 'bullish' or 'bearish'
    confidence = Column(String)  # 'Low', 'Medium', 'High'
    prediction_score = Column(Float)  # 0-100
    factors = Column(String)  # Comma-separated reasons for prediction
    
    created_at = Column(DateTime, default=datetime.now)
    def __repr__(self):
        return f"<StockPrediction(symbol='{self.symbol}', direction='{self.direction}')>"
