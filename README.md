# �� PredictiveStocks

> **Note:** As of 2025, the Yahoo Finance API now requires a paid subscription for reliable access. Free access may be limited, rate-limited, or unavailable, which can affect the functionality of this app.

## AI-Driven Stock Price Predictions

PredictiveStocks is an advanced stock price forecasting application powered by machine learning, specifically using LSTM models. Built with Python and Streamlit, this app aims to forecast stock prices and support investors in making informed decisions.

---

## 🌍 How It Works

PredictiveStocks is designed with the following key technologies and tools:

- **Streamlit** - For developing the interactive web application interface
- **FastAPI** - For creating high-performance, asynchronous APIs that handle data retrieval and processing
- **YFinance** - To retrieve financial data from Yahoo Finance
- **TensorFlow/Keras** - To build and train the LSTM time series forecasting model
- **Plotly** - To generate interactive charts and visualizations

## ✨ Key Features

- **Real-time Data** - Fetch the latest prices and fundamental information
- **Financial Charts** - Explore interactive historical and forecast charts
- **LSTM Forecasting** - Generate statistically robust stock price predictions
- **Model Tuning** - Optimize forecasting accuracy by adjusting the number of epochs and period length

## 🚀 Getting Started

### Local Installation

1. Clone the repo
    ```bash
    git clone https://github.com/MY_CURRENT_USERNAME/TrendTracker
    ```
2. Change directory
    ```bash
    cd TrendTracker
    ```
3. Install requirements
    ```bash
    pip install -r requirements.txt
    ```
4. Host the server
    ```bash
    uvicorn forecast:app --reload
    ```
5. Run the app
    ```bash
    streamlit run app.py
    ```

The app will be live at [http://localhost:8501](http://localhost:8501) and can also be found at: [https://trendtracker.streamlit.app](https://trendtracker.streamlit.app)

## 💡 Future Roadmap

Some potential features for future releases:

- **Advanced Quantitative Trading Strategies** - Utilize more complex algorithms for better market predictions.
- **In-Depth Fundamental Analysis** - Access extensive fundamental data for more informed decisions.
- **User Authentication and Account Management** - Secure and personalized user experience.
- **Customizable Alerts and Notifications** - Stay informed with tailored alerts for market changes.
- **Interactive Data Visualizations** - Gain insights through dynamic and interactive charts.

---

**⚖️ Disclaimer:** This application is for informational purposes only and does not constitute financial advice. The forecasts provided by the LSTM models are based on historical data and may not accurately predict future market movements. Always conduct your own research before trading.