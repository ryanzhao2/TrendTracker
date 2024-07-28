from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import json

app = FastAPI()

# Define a request model using Pydantic
class StockRequest(BaseModel):
    stock_name: str
    epochs: int

# Define the prediction endpoint
@app.post('/LSTM_Predict')
async def predict(stock_request: StockRequest):
    stock_name = stock_request.stock_name
    epochs = stock_request.epochs
    
    try:
        # Fetch historical stock data using yfinance
        df = yf.download(stock_name, start="2020-01-01", end="2022-12-31")
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for stock: {stock_name}")
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
        for epoch in range(epochs):
            # Train the model for one epoch
            history = model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=1)
            # Yield training progress
            yield json.dumps({
                'epoch': epoch + 1,
                'loss': float(history.history['loss'][0]),
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

        # Yield predictions
        yield json.dumps({
            'train_predictions': scaler.inverse_transform(model.predict(x_train)).tolist(),  # Predictions on training data
            'test_predictions': predict_price,  # Predictions on test data
            'future_predictions': predict_price[-90:]  # Future predictions (example: last 90 days)
        }) + "\n"

    # Return a StreamingResponse to stream results to the client
    return StreamingResponse(generate(), media_type="application/json")
