#!/usr/bin/env python3
"""
Test script for TrendTracker database integration
This script tests the database functionality to ensure everything is working correctly.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_database_connection():
    """Test database connection"""
    try:
        from database import DatabaseManager
        db_manager = DatabaseManager()
        print("✅ Database connection successful")
        return db_manager
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def test_stock_data_storage(db_manager):
    """Test storing and retrieving stock data"""
    try:
        # Create sample stock data
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        sample_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Volume': [1000000] * 10
        }, index=dates)
        
        # Store data
        success = db_manager.store_stock_data('TEST', sample_data)
        if success:
            print("✅ Stock data storage successful")
        else:
            print("❌ Stock data storage failed")
            return False
        
        # Retrieve data
        retrieved_data = db_manager.get_stock_data('TEST')
        if not retrieved_data.empty:
            print("✅ Stock data retrieval successful")
            print(f"   Retrieved {len(retrieved_data)} records")
        else:
            print("❌ Stock data retrieval failed")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Stock data test failed: {e}")
        return False

def test_prediction_storage(db_manager):
    """Test storing and retrieving predictions"""
    try:
        # Store a sample prediction
        success = db_manager.store_prediction(
            symbol='TEST',
            prediction_date=datetime.now(),
            predicted_price=150.50,
            confidence_score=0.85,
            model_version='v1.0',
            epochs_used=10,
            training_data_points=1000
        )
        
        if success:
            print("✅ Prediction storage successful")
        else:
            print("❌ Prediction storage failed")
            return False
        
        # Retrieve predictions
        predictions = db_manager.get_predictions('TEST', limit=5)
        if predictions:
            print("✅ Prediction retrieval successful")
            print(f"   Retrieved {len(predictions)} predictions")
        else:
            print("❌ Prediction retrieval failed")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def test_stock_info_storage(db_manager):
    """Test storing and retrieving stock information"""
    try:
        # Sample stock info
        stock_info = {
            'longName': 'Test Company Inc.',
            'sector': 'Technology',
            'industry': 'Software',
            'marketCap': 1000000000,
            'trailingPE': 25.5,
            'dividendYield': 0.02,
            'beta': 1.2,
            'priceToBook': 3.5,
            'debtToEquity': 0.5,
            'returnOnEquity': 0.15,
            'revenueGrowth': 0.10,
            'earningsGrowth': 0.08
        }
        
        # Store stock info
        success = db_manager.store_stock_info('TEST', stock_info)
        if success:
            print("✅ Stock info storage successful")
        else:
            print("❌ Stock info storage failed")
            return False
        
        # Retrieve stock info
        retrieved_info = db_manager.get_stock_info('TEST')
        if retrieved_info:
            print("✅ Stock info retrieval successful")
            print(f"   Company: {retrieved_info.company_name}")
        else:
            print("❌ Stock info retrieval failed")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Stock info test failed: {e}")
        return False

def test_watchlist_functionality(db_manager):
    """Test watchlist functionality"""
    try:
        # Add to watchlist
        success = db_manager.add_to_watchlist('test_user', 'TEST', 'Test stock for testing')
        if success:
            print("✅ Watchlist addition successful")
        else:
            print("❌ Watchlist addition failed")
            return False
        
        # Get watchlist
        watchlist = db_manager.get_watchlist('test_user')
        if watchlist:
            print("✅ Watchlist retrieval successful")
            print(f"   Found {len(watchlist)} items in watchlist")
        else:
            print("❌ Watchlist retrieval failed")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Watchlist test failed: {e}")
        return False

def test_model_performance_storage(db_manager):
    """Test storing model performance metrics"""
    try:
        success = db_manager.store_model_performance(
            symbol='TEST',
            model_version='v1.0',
            accuracy_score=85.5,
            mse_score=0.025,
            mae_score=0.15,
            test_period_start=datetime.now() - timedelta(days=30),
            test_period_end=datetime.now()
        )
        
        if success:
            print("✅ Model performance storage successful")
        else:
            print("❌ Model performance storage failed")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Model performance test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 TrendTracker Database Test")
    print("=" * 40)
    
    # Test database connection
    db_manager = test_database_connection()
    if not db_manager:
        sys.exit(1)
    
    # Run all tests
    tests = [
        ("Stock Data Storage", test_stock_data_storage),
        ("Prediction Storage", test_prediction_storage),
        ("Stock Info Storage", test_stock_info_storage),
        ("Watchlist Functionality", test_watchlist_functionality),
        ("Model Performance Storage", test_model_performance_storage)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func(db_manager):
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Database integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check your database configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main() 