from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
from datetime import datetime, timedelta
from database import DatabaseManager

app = FastAPI()

# Initialize database manager
db_manager = DatabaseManager()

# Define a request model using Pydantic
class StockRequest(BaseModel):
    stock_name: str
    epochs: int
    
# Define the prediction endpoint
@app.post("/")
async def predict(stock_request: StockRequest):
    stock_name = stock_request.stock_name
    epochs = stock_request.epochs
    
    try:
        # Fetch historical stock data using yfinance
        df = yf.download(stock_name, start="2020-01-01", end="2022-12-31")
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for stock: {stock_name}")
        
        # Store stock data in database
        db_manager.store_stock_data(stock_name, df)
        
        # Store stock fundamental information
        try:
            stock_info = yf.Ticker(stock_name).info
            db_manager.store_stock_info(stock_name, stock_info)
        except Exception as e:
            print(f"Error storing stock info: {e}")
            
    except Exception as e:
        # Handle errors during data fetching
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")
    
    # Prepare data for training
    data = df.filter(['Close'])  # Use only the 'Close' prices
    dataset = data.values  # Convert to a NumPy array
    training_data_len = int(np.ceil(len(dataset) * .80))  # Use 80% of the data for training
    scaler = MinMaxScaler(feature_range=(0, 1))  # Normalize data to range (0, 1)
    scaled_data = scaler.fit_transform(dataset)  # Scale the dataset
    train_data = scaled_data[0:int(training_data_len), :]  # Training data

    # Create training data sequences
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)  # Convert to arrays
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Reshape for LSTM input

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))  # First LSTM layer
    model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
    model.add(LSTM(64, return_sequences=False))  # Second LSTM layer
    model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
    model.add(Dense(1, activation='linear'))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])  # Compile model

    # Generator function to stream training progress and predictions
    def generate():
        final_loss = 0
        for epoch in range(epochs):
            # Train the model for one epoch
            history = model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=1)
            final_loss = float(history.history['loss'][0])
            # Yield training progress
            yield json.dumps({
                'epoch': epoch + 1,
                'loss': final_loss,
                'mean_squared_error': float(history.history['mean_squared_error'][0])
            }) + "\n"

        # After training, make predictions
        test_data = scaled_data[training_data_len - 60:, :]  # Prepare test data
        if len(test_data) < 60:
            raise HTTPException(status_code=400, detail="Not enough data to make predictions.")

        x_test = []
        y_test = dataset[training_data_len:, :]  # Test data
        for i in range(60, len(test_data)):
            x_test.append(test_data[i - 60:i, 0])
        x_test = np.array(x_test)  # Convert to array
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Reshape for LSTM input

        # Ensure correct lengths
        predictions = model.predict(x_test)  # Predict using the model
        predictions = scaler.inverse_transform(predictions)  # Inverse transform to original scale
        predict_price = [float(price[0]) for price in predictions]  # Convert predictions to list

        # Calculate model performance metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        # Calculate accuracy (percentage of predictions within 5% of actual)
        accuracy = np.mean(np.abs((predictions - y_test) / y_test) < 0.05) * 100
        
        # Store model performance in database
        test_period_start = df.index[training_data_len]
        test_period_end = df.index[-1]
        db_manager.store_model_performance(
            symbol=stock_name,
            model_version="v1.0",
            accuracy_score=accuracy,
            mse_score=mse,
            mae_score=mae,
            test_period_start=test_period_start,
            test_period_end=test_period_end
        )

        # Store predictions in database
        for i, pred_price in enumerate(predictions):
            prediction_date = df.index[training_data_len + i]
            db_manager.store_prediction(
                symbol=stock_name,
                prediction_date=prediction_date,
                predicted_price=float(pred_price[0]),
                confidence_score=1.0 - (mse / (df['Close'].max() ** 2)),  # Simple confidence score
                model_version="v1.0",
                epochs_used=epochs,
                training_data_points=len(x_train)
            )

        # Yield predictions
        yield json.dumps({
            'train_predictions': scaler.inverse_transform(model.predict(x_train)).tolist(),  # Predictions on training data
            'test_predictions': predict_price,  # Predictions on test data
            'future_predictions': predict_price[-90:],  # Future predictions (example: last 90 days)
            'model_performance': {
                'accuracy': accuracy,
                'mse': mse,
                'mae': mae,
                'final_loss': final_loss
            }
        }) + "\n"

    # Return a StreamingResponse to stream results to the client
    return StreamingResponse(generate(), media_type="application/json")

# New endpoint to get stored predictions
@app.get("/predictions/{symbol}")
async def get_predictions(symbol: str, limit: int = 100):
    """Get stored predictions for a stock"""
    predictions = db_manager.get_predictions(symbol, limit)
    return {
        "symbol": symbol,
        "predictions": [
            {
                "prediction_date": pred.prediction_date.isoformat(),
                "predicted_price": pred.predicted_price,
                "confidence_score": pred.confidence_score,
                "model_version": pred.model_version,
                "created_at": pred.created_at.isoformat()
            }
            for pred in predictions
        ]
    }

# New endpoint to get model performance
@app.get("/performance/{symbol}")
async def get_model_performance(symbol: str):
    """Get model performance metrics for a stock"""
    from database import ModelPerformance
    from database import SessionLocal
    
    session = SessionLocal()
    try:
        performance = session.query(ModelPerformance)\
            .filter(ModelPerformance.symbol == symbol)\
            .order_by(ModelPerformance.training_date.desc())\
            .first()
        
        if performance:
            return {
                "symbol": symbol,
                "model_version": performance.model_version,
                "accuracy_score": performance.accuracy_score,
                "mse_score": performance.mse_score,
                "mae_score": performance.mae_score,
                "training_date": performance.training_date.isoformat(),
                "test_period_start": performance.test_period_start.isoformat(),
                "test_period_end": performance.test_period_end.isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"No performance data found for {symbol}")
    finally:
        session.close()

# New endpoint to get stock data from database
@app.get("/stock-data/{symbol}")
async def get_stock_data(symbol: str, start_date: str = None, end_date: str = None):
    """Get historical stock data from database"""
    start_dt = None
    end_dt = None
    
    if start_date:
        start_dt = datetime.fromisoformat(start_date)
    if end_date:
        end_dt = datetime.fromisoformat(end_date)
    
    data = db_manager.get_stock_data(symbol, start_dt, end_dt)
    
    if data.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
    
    return {
        "symbol": symbol,
        "data": data.reset_index().to_dict('records')
    }

# New endpoint to get stock fundamental information
@app.get("/stock-info/{symbol}")
async def get_stock_info(symbol: str):
    """Get stock fundamental information from database"""
    stock_info = db_manager.get_stock_info(symbol)
    
    if not stock_info:
        raise HTTPException(status_code=404, detail=f"No information found for {symbol}")
    
    return {
        "symbol": stock_info.symbol,
        "company_name": stock_info.company_name,
        "sector": stock_info.sector,
        "industry": stock_info.industry,
        "market_cap": stock_info.market_cap,
        "pe_ratio": stock_info.pe_ratio,
        "dividend_yield": stock_info.dividend_yield,
        "beta": stock_info.beta,
        "price_to_book": stock_info.price_to_book,
        "debt_to_equity": stock_info.debt_to_equity,
        "return_on_equity": stock_info.return_on_equity,
        "revenue_growth": stock_info.revenue_growth,
        "earnings_growth": stock_info.earnings_growth,
        "updated_at": stock_info.updated_at.isoformat()
    }
