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
    zip_code = Column(String, nullable=True)
    country = Column(String, nullable=True)
    time_zone = Column(String, nullable=True)
    bio = Column(String, nullable=True)
    subscription_status = Column(String, default="Free")  # Example: Free, Premium
    last_login = Column(String, nullable=True)

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
# class Course(Base):
#     __tablename__ = "courses"

#     id = Column(Integer, primary_key=True, index=True)
#     title = Column(String, nullable=False)
#     description = Column(String, nullable=False)
#     level = Column(String, nullable=False)  # Beginner, Intermediate, Advanced
#     thumbnail = Column(String, nullable=True)
#     total_modules = Column(Integer, nullable=False)
#     duration_minutes = Column(Integer, nullable=False)
#     is_featured = Column(Boolean, default=False)
#     created_at = Column(DateTime(timezone=True), server_default=func.now())
    
#     # Relationships
#     modules = relationship("Module", back_populates="course")
#     enrollments = relationship("UserCourseProgress", back_populates="course")

# class Module(Base):
#     __tablename__ = "modules"

#     id = Column(Integer, primary_key=True, index=True)
#     course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
#     title = Column(String, nullable=False)
#     description = Column(String, nullable=True)
#     order_index = Column(Integer, nullable=False)  # For ordering modules within a course
#     duration_minutes = Column(Integer, nullable=False)
#     content_url = Column(String, nullable=True)  # URL or path to content
    
#     # Relationships
#     course = relationship("Course", back_populates="modules")
#     user_progress = relationship("UserModuleProgress", back_populates="module")

# class UserCourseProgress(Base):
#     __tablename__ = "user_course_progress"

#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
#     course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
#     enrollment_date = Column(DateTime(timezone=True), server_default=func.now())
#     last_accessed = Column(DateTime(timezone=True), server_default=func.now())
#     completed = Column(Boolean, default=False)
#     completion_date = Column(DateTime(timezone=True), nullable=True)
    
#     # Relationships
#     user = relationship("User")
#     course = relationship("Course", back_populates="enrollments")
#     module_progress = relationship("UserModuleProgress", back_populates="course_progress")

# class UserModuleProgress(Base):
#     __tablename__ = "user_module_progress"

#     id = Column(Integer, primary_key=True, index=True)
#     user_course_progress_id = Column(Integer, ForeignKey("user_course_progress.id"), nullable=False)
#     module_id = Column(Integer, ForeignKey("modules.id"), nullable=False)
#     completed = Column(Boolean, default=False)
#     completion_date = Column(DateTime(timezone=True), nullable=True)
    
#     # Relationships
#     course_progress = relationship("UserCourseProgress", back_populates="module_progress")
#     module = relationship("Module", back_populates="user_progress")

# class LearningStreak(Base):
#     __tablename__ = "learning_streaks"
    
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
#     streak_count = Column(Integer, default=0)
#     last_activity_date = Column(DateTime(timezone=True), server_default=func.now())
    
#     # Relationship
#     user = relationship("User")