# 📈 TrendTracker

> **Note:** As of 2025, the Yahoo Finance API now requires a paid subscription for reliable access. Free access may be limited, rate-limited, or unavailable, which can affect the functionality of this app.

## AI-Driven Stock Price Predictions with PostgreSQL

TrendTracker is an advanced stock price forecasting application powered by machine learning, specifically using LSTM models. Built with Python, Streamlit, and PostgreSQL, this app aims to forecast stock prices and support investors in making informed decisions.

---

## 🌍 How It Works

TrendTracker is designed with the following key technologies and tools:

- **Streamlit** - For developing the interactive web application interface
- **FastAPI** - For creating high-performance, asynchronous APIs that handle data retrieval and processing
- **PostgreSQL** - For storing historical stock data, predictions, and user watchlists
- **SQLAlchemy** - Python SQL toolkit and Object-Relational Mapping (ORM) library
- **YFinance** - To retrieve financial data from Yahoo Finance
- **TensorFlow/Keras** - To build and train the LSTM time series forecasting model
- **Plotly** - To generate interactive charts and visualizations

## ✨ Key Features

- **Real-time Data** - Fetch the latest prices and fundamental information
- **Financial Charts** - Explore interactive historical and forecast charts
- **LSTM Forecasting** - Generate statistically robust stock price predictions
- **Model Tuning** - Optimize forecasting accuracy by adjusting the number of epochs and period length
- **Database Storage** - Store and retrieve predictions, historical data, and user preferences
- **Watchlist Management** - Create and manage personalized stock watchlists
- **Model Performance Tracking** - Monitor and analyze prediction accuracy over time
- **Data Caching** - Improved performance through intelligent data caching

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- PostgreSQL database server
- pip package manager

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
4. Set up PostgreSQL database
    ```bash
    # Create a PostgreSQL database named 'trendtracker'
    createdb trendtracker
    ```
5. Configure database connection
    ```bash
    # Copy the example environment file
    cp env_example.txt .env
    # Edit .env file with your PostgreSQL credentials
    # DATABASE_URL=postgresql://username:password@localhost:5432/trendtracker
    ```
6. Initialize database
    ```bash
    python setup_database.py
    ```
7. Host the server
    ```bash
    uvicorn forecast:app --reload
    ```
8. Run the app
    ```bash
    streamlit run app.py
    ```

The app will be live at [http://localhost:8501](http://localhost:8501) and can also be found at: [https://trendtracker.streamlit.app](https://trendtracker.streamlit.app)

## 📊 Database Schema

The application uses PostgreSQL with the following main tables:

- **stock_data** - Historical stock price data
- **stock_predictions** - LSTM model predictions
- **stock_info** - Fundamental stock information
- **user_watchlists** - User watchlist management
- **model_performance** - Model accuracy and performance metrics

## 🔧 API Endpoints

The FastAPI backend provides the following endpoints:

- `POST /` - Generate stock price predictions
- `GET /predictions/{symbol}` - Get stored predictions for a stock
- `GET /performance/{symbol}` - Get model performance metrics
- `GET /stock-data/{symbol}` - Get historical stock data
- `GET /stock-info/{symbol}` - Get stock fundamental information

## 💡 Future Roadmap

Some potential features for future releases:

- **Advanced Quantitative Trading Strategies** - Utilize more complex algorithms for better market predictions
- **In-Depth Fundamental Analysis** - Access extensive fundamental data for more informed decisions
- **User Authentication and Account Management** - Secure and personalized user experience
- **Customizable Alerts and Notifications** - Stay informed with tailored alerts for market changes
- **Interactive Data Visualizations** - Gain insights through dynamic and interactive charts
- **Portfolio Management** - Track and analyze investment portfolios
- **Backtesting Framework** - Test trading strategies on historical data
- **Real-time Market Data** - Integrate with real-time market data providers

## 🛠️ Development

### Database Migrations

To create and run database migrations:

```bash
# Create a new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback migrations
alembic downgrade -1
```

### Environment Variables

Create a `.env` file with the following variables:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/trendtracker
API_URL=https://trendtracker-uvqu.onrender.com
DEBUG=True
ENVIRONMENT=development
```

---

**⚖️ Disclaimer:** This application is for informational purposes only and does not constitute financial advice. The forecasts provided by the LSTM models are based on historical data and may not accurately predict future market movements. Always conduct your own research before trading.