import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://username:password@localhost:5432/trendtracker')

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

class StockData(Base):
    """Model for storing historical stock data"""
    __tablename__ = "stock_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    date = Column(DateTime, index=True)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class StockPrediction(Base):
    """Model for storing LSTM predictions"""
    __tablename__ = "stock_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    prediction_date = Column(DateTime, index=True)
    predicted_price = Column(Float)
    confidence_score = Column(Float)
    model_version = Column(String)
    epochs_used = Column(Integer)
    training_data_points = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class StockInfo(Base):
    """Model for storing stock fundamental information"""
    __tablename__ = "stock_info"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True)
    company_name = Column(String)
    sector = Column(String)
    industry = Column(String)
    market_cap = Column(Float)
    pe_ratio = Column(Float)
    dividend_yield = Column(Float)
    beta = Column(Float)
    price_to_book = Column(Float)
    debt_to_equity = Column(Float)
    return_on_equity = Column(Float)
    revenue_growth = Column(Float)
    earnings_growth = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow)

class UserWatchlist(Base):
    """Model for storing user watchlists"""
    __tablename__ = "user_watchlists"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    symbol = Column(String, index=True)
    added_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

class ModelPerformance(Base):
    """Model for storing model performance metrics"""
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    model_version = Column(String)
    accuracy_score = Column(Float)
    mse_score = Column(Float)
    mae_score = Column(Float)
    training_date = Column(DateTime, default=datetime.utcnow)
    test_period_start = Column(DateTime)
    test_period_end = Column(DateTime)

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class DatabaseManager:
    """Database manager class for handling all database operations"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def store_stock_data(self, symbol: str, data: pd.DataFrame):
        """Store historical stock data"""
        session = self.SessionLocal()
        try:
            # Convert DataFrame to database records
            for index, row in data.iterrows():
                stock_data = StockData(
                    symbol=symbol,
                    date=index,
                    open_price=row.get('Open', 0),
                    high_price=row.get('High', 0),
                    low_price=row.get('Low', 0),
                    close_price=row.get('Close', 0),
                    volume=row.get('Volume', 0)
                )
                session.add(stock_data)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error storing stock data: {e}")
            return False
        finally:
            session.close()
    
    def store_prediction(self, symbol: str, prediction_date: datetime, 
                        predicted_price: float, confidence_score: float = None,
                        model_version: str = "v1.0", epochs_used: int = None,
                        training_data_points: int = None):
        """Store LSTM prediction"""
        session = self.SessionLocal()
        try:
            prediction = StockPrediction(
                symbol=symbol,
                prediction_date=prediction_date,
                predicted_price=predicted_price,
                confidence_score=confidence_score,
                model_version=model_version,
                epochs_used=epochs_used,
                training_data_points=training_data_points
            )
            session.add(prediction)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error storing prediction: {e}")
            return False
        finally:
            session.close()
    
    def store_stock_info(self, symbol: str, stock_info: dict):
        """Store stock fundamental information"""
        session = self.SessionLocal()
        try:
            # Check if stock info already exists
            existing = session.query(StockInfo).filter(StockInfo.symbol == symbol).first()
            
            if existing:
                # Update existing record
                existing.company_name = stock_info.get('longName', existing.company_name)
                existing.sector = stock_info.get('sector', existing.sector)
                existing.industry = stock_info.get('industry', existing.industry)
                existing.market_cap = stock_info.get('marketCap', existing.market_cap)
                existing.pe_ratio = stock_info.get('trailingPE', existing.pe_ratio)
                existing.dividend_yield = stock_info.get('dividendYield', existing.dividend_yield)
                existing.beta = stock_info.get('beta', existing.beta)
                existing.price_to_book = stock_info.get('priceToBook', existing.price_to_book)
                existing.debt_to_equity = stock_info.get('debtToEquity', existing.debt_to_equity)
                existing.return_on_equity = stock_info.get('returnOnEquity', existing.return_on_equity)
                existing.revenue_growth = stock_info.get('revenueGrowth', existing.revenue_growth)
                existing.earnings_growth = stock_info.get('earningsGrowth', existing.earnings_growth)
                existing.updated_at = datetime.utcnow()
            else:
                # Create new record
                new_stock_info = StockInfo(
                    symbol=symbol,
                    company_name=stock_info.get('longName'),
                    sector=stock_info.get('sector'),
                    industry=stock_info.get('industry'),
                    market_cap=stock_info.get('marketCap'),
                    pe_ratio=stock_info.get('trailingPE'),
                    dividend_yield=stock_info.get('dividendYield'),
                    beta=stock_info.get('beta'),
                    price_to_book=stock_info.get('priceToBook'),
                    debt_to_equity=stock_info.get('debtToEquity'),
                    return_on_equity=stock_info.get('returnOnEquity'),
                    revenue_growth=stock_info.get('revenueGrowth'),
                    earnings_growth=stock_info.get('earningsGrowth')
                )
                session.add(new_stock_info)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error storing stock info: {e}")
            return False
        finally:
            session.close()
    
    def get_stock_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None):
        """Retrieve historical stock data"""
        session = self.SessionLocal()
        try:
            query = session.query(StockData).filter(StockData.symbol == symbol)
            
            if start_date:
                query = query.filter(StockData.date >= start_date)
            if end_date:
                query = query.filter(StockData.date <= end_date)
            
            results = query.order_by(StockData.date).all()
            
            # Convert to DataFrame
            data = []
            for result in results:
                data.append({
                    'Date': result.date,
                    'Open': result.open_price,
                    'High': result.high_price,
                    'Low': result.low_price,
                    'Close': result.close_price,
                    'Volume': result.volume
                })
            
            return pd.DataFrame(data).set_index('Date')
        except Exception as e:
            print(f"Error retrieving stock data: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def get_predictions(self, symbol: str, limit: int = 100):
        """Retrieve predictions for a stock"""
        session = self.SessionLocal()
        try:
            results = session.query(StockPrediction)\
                .filter(StockPrediction.symbol == symbol)\
                .order_by(StockPrediction.prediction_date.desc())\
                .limit(limit)\
                .all()
            
            return results
        except Exception as e:
            print(f"Error retrieving predictions: {e}")
            return []
        finally:
            session.close()
    
    def get_stock_info(self, symbol: str):
        """Retrieve stock fundamental information"""
        session = self.SessionLocal()
        try:
            result = session.query(StockInfo).filter(StockInfo.symbol == symbol).first()
            return result
        except Exception as e:
            print(f"Error retrieving stock info: {e}")
            return None
        finally:
            session.close()
    
    def add_to_watchlist(self, user_id: str, symbol: str, notes: str = None):
        """Add stock to user watchlist"""
        session = self.SessionLocal()
        try:
            # Check if already in watchlist
            existing = session.query(UserWatchlist)\
                .filter(UserWatchlist.user_id == user_id, UserWatchlist.symbol == symbol)\
                .first()
            
            if not existing:
                watchlist_item = UserWatchlist(
                    user_id=user_id,
                    symbol=symbol,
                    notes=notes
                )
                session.add(watchlist_item)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            print(f"Error adding to watchlist: {e}")
            return False
        finally:
            session.close()
    
    def get_watchlist(self, user_id: str):
        """Get user watchlist"""
        session = self.SessionLocal()
        try:
            results = session.query(UserWatchlist)\
                .filter(UserWatchlist.user_id == user_id)\
                .order_by(UserWatchlist.added_at.desc())\
                .all()
            return results
        except Exception as e:
            print(f"Error retrieving watchlist: {e}")
            return []
        finally:
            session.close()
    
    def store_model_performance(self, symbol: str, model_version: str, 
                               accuracy_score: float, mse_score: float, mae_score: float,
                               test_period_start: datetime, test_period_end: datetime):
        """Store model performance metrics"""
        session = self.SessionLocal()
        try:
            performance = ModelPerformance(
                symbol=symbol,
                model_version=model_version,
                accuracy_score=accuracy_score,
                mse_score=mse_score,
                mae_score=mae_score,
                test_period_start=test_period_start,
                test_period_end=test_period_end
            )
            session.add(performance)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error storing model performance: {e}")
            return False
        finally:
            session.close()

# Initialize database
init_db() 