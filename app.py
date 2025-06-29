import time
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf
import datetime
import numpy as np
import os
import json
from datetime import datetime, timedelta

from helper import *
from database import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager()

# Constants
#API_URL is the API endpoint to get LSTM predictions
API_URL = "https://trendtracker-uvqu.onrender.com"

MIN_DATE = datetime.date(2020, 1, 1)
MAX_DATE = datetime.date(2022, 12, 31)
DATA_DIR = 'data'

def ensure_data_directory():
    """Ensure the data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def load_tickers():
    # Load tickers from CSV
    file_name = os.path.join(DATA_DIR, 'nasdaq_screener.csv')
    tickers_df = pd.read_csv(file_name)
    return tickers_df["Symbol"].tolist()

def home_page():
    st.markdown("""
        <style>
            .title { font-size: 48px; font-weight: bold; text-align: center; margin-bottom: 10px; }
            .subtitle { font-size: 36px; text-align: center; margin-bottom: 40px; }
            .section-title { font-size: 28px; margin-top: 40px; margin-bottom: 20px; }
            .section-title2 { font-size: 24px; margin-top: 40px; margin-bottom: 20px; }
            .text { font-size: 18px; margin-bottom: 20px; line-height: 1.6; }
            .list-item { font-size: 18px; margin-bottom: 10px; line-height: 1.6; }
            .code-block { background-color: #ECF0F1; padding: 10px; font-size: 16px; border-radius: 5px; margin-bottom: 20px; }
            .footer { font-size: 16px; text-align: center; margin-top: 40px; }
        </style>
        """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="title">📈 TrendTracker</div>
        <div class="subtitle">AI-Driven Stock Price Predictions</div>
        
        <div class="text">
            TrendTracker is an advanced stock price forecasting application powered by machine learning, specifically using LSTM models. Built with Python, Streamlit, and PostgreSQL, this app aims to forecast stock prices and support investors in making informed decisions.
        </div>""", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown(
        """        
        <div class="section-title">🌍 How It Works</div>
        <div class="text">
            TrendTracker is designed with the following key technologies and tools:
        </div>
        <ul>
            <li class="list-item">Streamlit - For developing the interactive web application interface</li>
            <li class="list-item">FastAPI - For creating high-performance, asynchronous APIs that handle data retrieval and processing</li>
            <li class="list-item">PostgreSQL - For storing historical stock data, predictions, and user watchlists</li>
            <li class="list-item">YFinance - To retrieve financial data from Yahoo Finance</li>
            <li class="list-item">TensorFlow/Keras - To build and train the LSTM time series forecasting model</li>
            <li class="list-item">Plotly - To generate interactive charts and visualizations</li>
        </ul>
        
        <div class="section-title">✨ Key Features</div>
        <ul>
            <li class="list-item"><strong>Real-time Data</strong> - Fetch the latest prices and fundamental information</li>
            <li class="list-item"><strong>Financial Charts</strong> - Explore interactive historical and forecast charts</li>
            <li class="list-item"><strong>LSTM Forecasting</strong> - Generate statistically robust stock price predictions</li>
            <li class="list-item"><strong>Model Tuning</strong> - Optimize forecasting accuracy by adjusting the number of epochs and period length</li>
            <li class="list-item"><strong>Database Storage</strong> - Store and retrieve predictions, historical data, and user preferences</li>
            <li class="list-item"><strong>Watchlist Management</strong> - Create and manage personalized stock watchlists</li>
        </ul>
        
        <div class="section-title">🚀 Getting Started</div>
        <div class="section-title2">Local Installation</div>
        <ol>
            <li class="list-item">Clone the repo</li>
            <pre class="code-block"><code>git clone https://github.com/MY_CURRENT_USERNAME/TrendTracker.git</code></pre>
            <li class="list-item">Change directory</li>
            <pre class="code-block"><code>cd TrendTracker</code></pre>
            <li class="list-item">Install requirements</li>
            <pre class="code-block"><code>pip install -r requirements.txt</code></pre>
            <li class="list-item">Set up PostgreSQL database and configure DATABASE_URL in .env file</li>
            <li class="list-item">Host the server</li>
            <pre class="code-block"><code>uvicorn forecast:app --reload</code></pre>
            <li class="list-item">Run the app</li>
            <pre class="code-block"><code>streamlit run app.py</code></pre>
        </ol>
        <div class="text">
            The app will be live at <code>http://localhost:8501</code>
        </div>
        
        <div class="section-title">💡 Future Roadmap</div>
        <div class="text">Some potential features for future releases:</div>
        <ul>
            <li class="list-item"><strong>Advanced Quantitative Trading Strategies</strong> - Utilize more complex algorithms for improved forecasts.</li>
            <li class="list-item"><strong>In-Depth Fundamental Analysis</strong> - Access extensive fundamental data for more informed decisions.</li>
            <li class="list-item"><strong>User Authentication and Account Management</strong> - Secure and personalized user experience.</li>
            <li class="list-item"><strong>Customizable Alerts and Notifications</strong> - Stay informed with tailored alerts for market changes.</li>
            <li class="list-item"><strong>Interactive Data Visualizations</strong> - Gain insights through dynamic and interactive charts.</li>
        </ul>
        
        <div class="footer">
            <strong>⚖️ Disclaimer:</strong> This application is for informational purposes only and does not constitute financial advice. 
            The forecasts provided by the LSTM models are based on historical data and may not accurately predict future market movements. 
            Always conduct your own research before trading.
        </div>
        """,
        unsafe_allow_html=True
    )

def stock_info():
    st.title("Stock Information")
    st.write("This page provides stock information and historical data.")
    
    st.sidebar.markdown("## **User Input Features**")
    
    # Fetch stock tickers from CSV
    stock_tickers = load_tickers()
    
    stock = st.sidebar.selectbox("Choose a stock", stock_tickers)
    
    if stock:
        stock_data_info = display_stock_details(stock)
        
        if stock_data_info:
            # Display model performance if available
            performance = get_model_performance(stock)
            if performance:
                st.markdown("## **Model Performance**")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{performance['accuracy_score']:.2f}%")
                col2.metric("MSE", f"{performance['mse_score']:.4f}")
                col3.metric("MAE", f"{performance['mae_score']:.4f}")
                col4.metric("Model Version", performance['model_version'])
                
                st.write(f"Last trained: {performance['training_date'].strftime('%Y-%m-%d %H:%M')}")
    else:
        st.write("Please select a stock from the dropdown.")

def forecast_page():
    ensure_data_directory()
    st.title("Stock Price Forecast")

    st.sidebar.markdown("## **User Input Features**")

    stock_tickers = load_tickers()
    selected_stock = st.sidebar.selectbox("Select Stock", stock_tickers)

    periods = fetch_periods_intervals()
    default_period = '1mo'
    default_interval = '1d'

    st.sidebar.markdown("### **Select period**")
    period = st.sidebar.selectbox("Choose a period", list(periods.keys()), index=list(periods.keys()).index(default_period) if default_period in periods.keys() else 0)

    st.sidebar.markdown("### **Select interval**")
    interval = st.sidebar.selectbox("Choose an interval", periods[period], index=periods[period].index(default_interval) if default_interval in periods[period] else 0)

    epochs = st.sidebar.slider("Select number of epochs", min_value=1, max_value=50, value=5, step=1)

    st.markdown("## **Historical Data**")

    if selected_stock:
        try:
            stock_data = fetch_stock_history(selected_stock, period, interval)

            if stock_data.empty:
                st.error(f"No data found for stock: {selected_stock} with period {period} and interval {interval}")
                return

            #Display historical data
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=stock_data.index,
                        open=stock_data["Open"],
                        high=stock_data["High"],
                        low=stock_data["Low"],
                        close=stock_data["Close"],
                    )
                ]
            )
            fig.update_layout(title = f'Historical data for: {selected_stock}',
                              xaxis_rangeslider_visible=False, yaxis=dict(
                        tickprefix='$',  # Add $ symbol before the price values
                        title='Price (USD)'  # Optional: set the title for the y-axis
                    )
                    )
            st.plotly_chart(fig, use_container_width=True)

            if st.button("Predict"):
                st.markdown("## **Stock Prediction**")
                init_message = st.empty()
                init_message.text("Initializing...")

                payload = {"stock_name": selected_stock, "epochs": epochs}

                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    response = requests.post(API_URL, json=payload, stream=True)
                    response.raise_for_status()

                    # Process streaming response
                    train_predictions = []
                    test_predictions = []
                    future_predictions = []
                    model_performance = {}

                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line.decode('utf-8'))
                            
                            if 'epoch' in data:
                                # Training progress
                                epoch = data['epoch']
                                loss = data['loss']
                                progress = epoch / epochs
                                progress_bar.progress(progress)
                                status_text.text(f"Training epoch {epoch}/{epochs} - Loss: {loss:.4f}")
                            
                            elif 'train_predictions' in data:
                                # Final predictions
                                train_predictions = data['train_predictions']
                                test_predictions = data['test_predictions']
                                future_predictions = data['future_predictions']
                                model_performance = data.get('model_performance', {})
                                
                                progress_bar.progress(1.0)
                                status_text.text("Prediction complete!")

                    # Display model performance
                    if model_performance:
                        st.markdown("### **Model Performance Metrics**")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", f"{model_performance.get('accuracy', 0):.2f}%")
                        col1.metric("MSE", f"{model_performance.get('mse', 0):.4f}")
                        col1.metric("MAE", f"{model_performance.get('mae', 0):.4f}")
                        col1.metric("Final Loss", f"{model_performance.get('final_loss', 0):.4f}")

                    # Create prediction chart
                    if test_predictions:
                        # Get dates for predictions
                        test_dates = stock_data.index[-len(test_predictions):]

                    fig = go.Figure()
                        
                        # Add actual prices
                        fig.add_trace(go.Scatter(
                            x=stock_data.index,
                            y=stock_data['Close'],
                            mode='lines',
                            name='Actual Price',
                            line=dict(color='blue')
                        ))
                        
                        # Add test predictions
                        fig.add_trace(go.Scatter(
                            x=test_dates,
                            y=test_predictions,
                            mode='lines',
                            name='Predicted Price',
                            line=dict(color='red', dash='dash')
                        ))
                        
                    fig.update_layout(
                            title=f'Stock Price Prediction for {selected_stock}',
                            xaxis_title='Date',
                            yaxis_title='Price (USD)',
                            yaxis=dict(tickprefix='$')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to prediction service: {e}")
                except json.JSONDecodeError as e:
                    st.error(f"Error parsing response: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

        except Exception as e:
            st.error(f"Error fetching stock data: {e}")

def watchlist_page():
    st.title("Watchlist Management")
    
    # Simple user ID (in a real app, this would come from authentication)
    user_id = st.session_state.get('user_id', 'default_user')
    
    st.markdown("## **Add to Watchlist**")
    
    stock_tickers = load_tickers()
    new_stock = st.selectbox("Select stock to add", stock_tickers)
    notes = st.text_area("Notes (optional)")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add to Watchlist"):
            if db_manager.add_to_watchlist(user_id, new_stock, notes):
                st.success(f"Added {new_stock} to watchlist!")
            else:
                st.warning(f"{new_stock} is already in your watchlist.")
    
    with col2:
        if st.button("Refresh Watchlist"):
            st.rerun()
    
    st.markdown("## **Your Watchlist**")
    
    watchlist_items = db_manager.get_watchlist(user_id)
    
    if watchlist_items:
        for item in watchlist_items:
            with st.expander(f"{item.symbol} - Added {item.added_at.strftime('%Y-%m-%d')}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Get current stock info
                    try:
                        stock_info = fetch_stock_info(item.symbol)
                        current_price = stock_info.get('regularMarketPrice', 'N/A')
                        company_name = stock_info.get('longName', 'N/A')
                        
                        st.write(f"**Company:** {company_name}")
                        st.write(f"**Current Price:** ${current_price}" if current_price != 'N/A' else f"**Current Price:** {current_price}")
                        
                        if item.notes:
                            st.write(f"**Notes:** {item.notes}")
                    except Exception as e:
                        st.write(f"Error fetching data for {item.symbol}: {e}")
                
                with col2:
                    if st.button(f"Remove {item.symbol}", key=f"remove_{item.symbol}"):
                        # Note: You would need to add a remove function to DatabaseManager
                        st.warning("Remove functionality not implemented yet")
    else:
        st.write("Your watchlist is empty. Add some stocks to get started!")

def database_stats_page():
    st.title("Database Statistics")
    
    try:
        from database import SessionLocal, StockData, StockPrediction, StockInfo, UserWatchlist, ModelPerformance
        
        session = SessionLocal()
        
        # Get statistics
        total_stocks = session.query(StockData.symbol).distinct().count()
        total_predictions = session.query(StockPrediction).count()
        total_stock_info = session.query(StockInfo).count()
        total_watchlist_items = session.query(UserWatchlist).count()
        total_performance_records = session.query(ModelPerformance).count()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Stocks", total_stocks)
        col2.metric("Total Predictions", total_predictions)
        col3.metric("Stock Info Records", total_stock_info)
        
        col1, col2 = st.columns(2)
        col1.metric("Watchlist Items", total_watchlist_items)
        col2.metric("Performance Records", total_performance_records)
        
        # Recent predictions
        st.markdown("## **Recent Predictions**")
        recent_predictions = session.query(StockPrediction)\
            .order_by(StockPrediction.created_at.desc())\
            .limit(10)\
            .all()
        
        if recent_predictions:
            pred_data = []
            for pred in recent_predictions:
                pred_data.append({
                    'Symbol': pred.symbol,
                    'Predicted Price': f"${pred.predicted_price:.2f}",
                    'Date': pred.prediction_date.strftime('%Y-%m-%d'),
                    'Confidence': f"{pred.confidence_score:.2f}" if pred.confidence_score else 'N/A',
                    'Created': pred.created_at.strftime('%Y-%m-%d %H:%M')
                })
            
            st.dataframe(pd.DataFrame(pred_data))
        else:
            st.write("No predictions found in database.")
        
        session.close()
        
    except Exception as e:
        st.error(f"Error accessing database: {e}")

def learn_more_page():
    st.title("Learn More")
    st.write("This page provides additional information about the project and its features.")
    
    st.markdown("## **About the Technology Stack**")
    
    st.markdown("### **Machine Learning**")
    st.write("""
    - **LSTM (Long Short-Term Memory)**: A type of recurrent neural network that can learn long-term dependencies
    - **TensorFlow/Keras**: Deep learning framework used for building and training the LSTM model
    - **Scikit-learn**: Used for data preprocessing and performance metrics calculation
    """)
    
    st.markdown("### **Web Framework**")
    st.write("""
    - **Streamlit**: Python library for creating web applications with minimal code
    - **FastAPI**: Modern, fast web framework for building APIs with Python
    """)
    
    st.markdown("### **Data Storage**")
    st.write("""
    - **PostgreSQL**: Relational database for storing historical stock data, predictions, and user data
    - **SQLAlchemy**: Python SQL toolkit and Object-Relational Mapping (ORM) library
    """)
    
    st.markdown("### **Data Sources**")
    st.write("""
    - **Yahoo Finance**: Primary source for real-time and historical stock data
    - **NASDAQ Screener**: Source for stock ticker symbols and basic company information
    """)
    
    st.markdown("## **Model Architecture**")
    st.write("""
    The LSTM model used in this application consists of:
    - **Input Layer**: 60 time steps of historical price data
    - **LSTM Layer 1**: 128 units with return sequences
    - **Dropout Layer**: 20% dropout for regularization
    - **LSTM Layer 2**: 64 units
    - **Dropout Layer**: 20% dropout for regularization
    - **Dense Layer**: 1 unit for price prediction
    """)
    
    st.markdown("## **Performance Metrics**")
    st.write("""
    The model performance is evaluated using:
    - **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values
    - **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values
    - **Accuracy**: Percentage of predictions within 5% of actual values
    """)

def main():
    # Set up session state to manage page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Choose a page",
        ['Home', 'Stock Information', 'Forecast', 'Watchlist', 'Database Stats', 'Learn More'],
        index=['Home', 'Stock Information', 'Forecast', 'Watchlist', 'Database Stats', 'Learn More'].index(st.session_state.current_page)
    )
    
    # Update session state
    st.session_state.current_page = page
    
    # Display selected page
    if page == 'Home':
        home_page()
    elif page == 'Stock Information':
        stock_info()
    elif page == 'Forecast':
        forecast_page()
    elif page == 'Watchlist':
        watchlist_page()
    elif page == 'Database Stats':
        database_stats_page()
    elif page == 'Learn More':
        learn_more_page()

if __name__ == "__main__":
    main()

