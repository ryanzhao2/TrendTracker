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

from helper import *

# Constants
#API_URL is the API endpoint to get LSTM predictions
API_URL = "http://127.0.0.1:8000/LSTM_Predict"
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
            .title { font-size: 48px; font-weight: bold; color: #E2EDF1; text-align: center; margin-bottom: 10px; }
            .subtitle { font-size: 36px; color: #E2EDF1; text-align: center; margin-bottom: 40px; }
            .section-title { font-size: 28px; color: #E2EDF1; margin-top: 40px; margin-bottom: 20px; }
            .section-title2 { font-size: 24px; color: #E2EDF1; margin-top: 40px; margin-bottom: 20px; }
            .text { font-size: 18px; color: #E2EDF1; margin-bottom: 20px; line-height: 1.6; }
            .list-item { font-size: 18px; color: #E2EDF1; margin-bottom: 10px; line-height: 1.6; }
            .code-block { background-color: #ECF0F1; padding: 10px; font-size: 16px; border-radius: 5px; margin-bottom: 20px; }
            .footer { font-size: 16px; color: #95A5A6; text-align: center; margin-top: 40px; }
        </style>
        """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="title">📈 TrendTracker</div>
        <div class="subtitle">AI-Driven Stock Price Predictions</div>
        
        <div class="text">
            TrendTracker is an advanced stock price forecasting application powered by machine learning, specifically using LSTM models. Built with Python and Streamlit, this app aims to forecast stock prices and support investors in making informed decisions.
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
        </ul>
        
        <div class="section-title">🚀 Getting Started</div>
        <div class="section-title2">Local Installation</div>
        <ol>
            <li class="list-item">Clone the repo</li>
            <pre class="code-block"><code>git clone https://github.com/user/stockastic.git</code></pre>
            <li class="list-item">Install requirements</li>
            <pre class="code-block"><code>pip install -r requirements.txt</code></pre>
            <li class="list-item">Change directory</li>
            <pre class="code-block"><code>cd streamlit_app</code></pre>
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
            for category, info in stock_data_info.items():
                st.markdown(f"## **{category}**")
                for key, value in info.items():
                    st.write(f"**{key}:** {value}")
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

                    train_predictions = []
                    test_predictions = []
                    future_predictions = []
                    total_epochs = epochs

                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line.decode('utf-8'))
                            if 'epoch' in data:
                                init_message.empty()
                                epoch = data.get('epoch', 0)
                                loss = data.get('loss', 0)
                                mse = data.get('mean_squared_error', 0)
                                progress = int((epoch / total_epochs) * 100)
                                progress_bar.progress(progress)
                                status_text.text(f"Processing... Epoch {epoch}/{total_epochs} | Loss: {loss:.5f} | MSE: {mse:.5f}")
                            elif 'train_predictions' in data and 'test_predictions' in data:
                                train_predictions = data.get("train_predictions", [])
                                test_predictions = data.get("test_predictions", [])
                                future_predictions = data.get("future_predictions", [])

                    progress_bar.empty()
                    status_text.empty()

                    if not train_predictions or not test_predictions:
                        st.warning("Predictions data is empty.")
                        return

                    # train_pred_dates = stock_data.index[:len(train_predictions)]
                    test_pred_dates = stock_data.index[-len(test_predictions):]
                    future_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=90)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Close"], name='Actual'))
                    # fig.add_trace(go.Scatter(x=train_pred_dates, y=train_predictions, name='Train Predictions'))
                    fig.add_trace(go.Scatter(x=test_pred_dates, y=test_predictions, name='Test Predictions'))
                    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, name='Future Predictions'))
                    fig.update_layout(
                    title=f"Stock Predictions for: {selected_stock}", yaxis=dict(
                        tickprefix='$',  # Add $ symbol before the price values
                        title='Price (USD)'  # Optional: set the title for the y-axis
                    )
                )   
                    st.plotly_chart(fig)

                except requests.exceptions.RequestException as e:
                    st.error(f"Error occurred while making the request: {e}")
                    progress_bar.empty()
                    status_text.empty()
                except KeyError as e:
                    st.error(f"Key error: {e}")
                    progress_bar.empty()
                    status_text.empty()
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    progress_bar.empty()
                    status_text.empty()
        except Exception as e:
            st.error(f"An error occurred: {e}")

def learn_more_page():
    st.title("More Information")
    st.write("This page provides additional resources and information about the Stock Predictor App, Long Short-Term Memory (LSTM) networks, and trading.")

    st.subheader("📚 Learn More About LSTM")
    st.markdown("""
    Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems. Here are some resources to help you learn more:

    - [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
    - [Stock Market Predictions with LSTM in Python](https://www.datacamp.com/tutorial/lstm-python-stock-market)
    - [Long Short-Term Memory (LSTM) Networks with TensorFlow](https://www.tensorflow.org/guide/keras/rnn)
    - [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurélien Géron (Book)

    """)
    st.markdown("---")

    st.subheader("📈 Learn More About Trading")
    st.markdown("""
    Trading involves buying and selling financial instruments to make a profit. Here are some resources to help you learn more:

    - [How to Trade Stocks: Six Steps to Get Started](https://www.investopedia.com/learn-how-to-trade-the-market-in-5-steps-4692230)
    - [Technical Analysis: What It Is and How To Use It in Investing](https://www.investopedia.com/terms/t/technicalanalysis.asp)
    - [Algorithmic Trading and DMA: An Introduction to Direct Access Trading Strategies](https://www.amazon.com/Algorithmic-Trading-DMA-introduction-strategies/dp/0956399207) by Barry Johnson (Book)
    - [Quantitative Trading: How to Build Your Own Algorithmic Trading Business](https://www.amazon.com/Quantitative-Trading-Build-Algorithmic-Business/dp/0470284889) by Ernie Chan (Book)
    - [Dancing With The Analysts- A Financial Novel](https://www.amazon.ca/Dancing-Analysts-Financial-Novel-Mallach/dp/0970568428) by David A Mallach (Book)
    """)

    st.markdown("---")

    st.subheader("🎥 Video Tutorials")
    st.markdown("""
    Here are some video tutorials to get you started with LSTM networks and trading strategies:

    - **Introduction to LSTM**:
      [![LSTM Networks](https://img.youtube.com/vi/8HyCNIVRbSU/0.jpg)](https://www.youtube.com/watch?v=8HyCNIVRbSU)
    
    - **Trading for Beginners**:
      [![Trading for Beginners](https://img.youtube.com/vi/_YVQN6_nkfs/0.jpg)](https://www.youtube.com/watch?v=_YVQN6_nkfs)
    """)

    st.subheader("🔗 Useful Links")
    st.markdown("""
    - [TensorFlow Documentation](https://www.tensorflow.org/guide/keras/rnn)
    - [Investopedia](https://www.investopedia.com/)
    - [Yahoo Finance](https://finance.yahoo.com/)
    """)

# Main function to handle navigation
def main():
    # Set up session state to manage page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    # Custom CSS for removing button outlines and styling
    st.markdown("""
        <style>
        .stButton button {
            border: none;
            background-color: #31333F;
            padding: 8px;
            margin: 5px 0;
            border-radius: 5px;
            font-size: 16px;
            color: white;
            cursor: pointer;
            width: 100%;
            text-align: center;
        }
        .stButton button:hover {
            background-color: #ddd;
        }
        </style>
        """, unsafe_allow_html=True)

    # Sidebar navigation with custom-styled buttons
    st.sidebar.title("Navigation")
    if st.sidebar.button("Home"):
        st.session_state.page = "Home"
    if st.sidebar.button("Stock Info"):
        st.session_state.page = "Stock Info"
    if st.sidebar.button("Forecast"):
        st.session_state.page = "Forecast"
    if st.sidebar.button("Learn More"):
        st.session_state.page = "Learn More"
        
    # Display the selected page
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Stock Info":
        stock_info()
    elif st.session_state.page == "Forecast":
        forecast_page()
    elif st.session_state.page == "Learn More":
        learn_more_page()

if __name__ == "__main__":
    main()

