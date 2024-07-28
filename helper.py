import streamlit as st
import pandas as pd
import yfinance as yf
from pathlib import Path
import datetime as dt
import os

def fetch_periods_intervals():
    # Create dictionary for periods and intervals
    periods = {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"],
    }

    # Return the dictionary
    return periods

# Function to fetch the stock history
def fetch_stock_history(stock_ticker, period, interval):
    # Pull the data for the first security
    stock_data = yf.Ticker(stock_ticker)

    # Extract full of the stock
    stock_data_history = stock_data.history(period=period, interval=interval)[
        ["Open", "High", "Low", "Close"]
    ]

    # Return the stock data
    return stock_data_history

# Function to fetch the stock info
def fetch_stock_info(stock_ticker):
    # Pull the data for the first security
    stock_data = yf.Ticker(stock_ticker)

    # Extract full of the stock
    stock_data_info = stock_data.info

    # Function to safely get value from dictionary or return "N/A"
    def safe_get(data_dict, key):
        return data_dict.get(key, "N/A")


# Function to display detailed stock information using Streamlit
def display_stock_details(stock_name):
    stock = yf.Ticker(stock_name)
    stock_info = stock.info

    stock_data_info = {

        "General Information": {
            "symbol": stock_info.get("symbol", None),
            "longName": stock_info.get("longName", None),
            "currency": stock_info.get("currency", None),
            "exchange": stock_info.get("exchange", None),
        },
        
        "Market Data": {
            "regularMarketPrice": stock_info.get("regularMarketPrice", None),
            "regularMarketOpen": stock_info.get("regularMarketOpen", None),
            "regularMarketDayLow": stock_info.get("regularMarketDayLow", None),
            "regularMarketDayHigh": stock_info.get("regularMarketDayHigh", None),
            "fiftyTwoWeekLow": stock_info.get("fiftyTwoWeekLow", None),
            "fiftyTwoWeekHigh": stock_info.get("fiftyTwoWeekHigh", None),
            "fiftyDayAverage": stock_info.get("fiftyDayAverage", None),
        },
        "Volume and Shares": {
            "volume": stock_info.get("volume", None),
            "regularMarketVolume": stock_info.get("regularMarketVolume", None),
            "averageVolume": stock_info.get("averageVolume", None),
            "averageVolume10days": stock_info.get("averageVolume10days", None),
            "averageDailyVolume10Day": stock_info.get("averageDailyVolume10Day", None),
            "sharesOutstanding": stock_info.get("sharesOutstanding", None),
            "impliedSharesOutstanding": stock_info.get("impliedSharesOutstanding", None),
            "floatShares": stock_info.get("floatShares", None),
        },
        "Dividends and Yield": {
            "dividendRate": stock_info.get("dividendRate", None),
            "dividendYield": stock_info.get("dividendYield", None),
            "payoutRatio": stock_info.get("payoutRatio", None),
        },
        "Valuation and Ratios": {
            "marketCap": stock_info.get("marketCap", None),
            "enterpriseValue": stock_info.get("enterpriseValue", None),
            "priceToBook": stock_info.get("priceToBook", None),
            "debtToEquity": stock_info.get("debtToEquity", None),
            "grossMargins": stock_info.get("grossMargins", None),
            "profitMargins": stock_info.get("profitMargins", None),
        },
        "Financial Performance": {
            "totalRevenue": stock_info.get("totalRevenue", None),
            "revenuePerShare": stock_info.get("revenuePerShare", None),
            "totalCash": stock_info.get("totalCash", None),
            "totalCashPerShare": stock_info.get("totalCashPerShare", None),
            "totalDebt": stock_info.get("totalDebt", None),
            "earningsGrowth": stock_info.get("earningsGrowth", None),
            "revenueGrowth": stock_info.get("revenueGrowth", None),
            "returnOnAssets": stock_info.get("returnOnAssets", None),
            "returnOnEquity": stock_info.get("returnOnEquity", None),
        },
        "Cash Flow": {
            "freeCashflow": stock_info.get("freeCashflow", None),
            "operatingCashflow": stock_info.get("operatingCashflow", None),
        },
        "Analyst Targets": {
            "targetHighPrice": stock_info.get("targetHighPrice", None),
            "targetLowPrice": stock_info.get("targetLowPrice", None),
            "targetMeanPrice": stock_info.get("targetMeanPrice", None),
            "targetMedianPrice": stock_info.get("targetMedianPrice", None),
        },

    }
    
    # Display stock information in Streamlit app
    st.markdown("## **General Information**")
    col1, col2 = st.columns(2)
    col1.dataframe(pd.DataFrame({"Symbol": [stock_data_info["General Information"]["symbol"]]}), hide_index=True, width=500)
    col2.dataframe(pd.DataFrame({"Long Name": [stock_data_info["General Information"]["longName"]]}), hide_index=True, width=500)
    
    col1, col2 = st.columns(2)
    col1.dataframe(pd.DataFrame({"Currency": [stock_data_info["General Information"]["currency"]]}), hide_index=True, width=500)
    col2.dataframe(pd.DataFrame({"Exchange": [stock_data_info["General Information"]["exchange"]]}), hide_index=True, width=500)

    st.markdown("## **Market Data**")
    col1, col2 = st.columns(2)
    col1.dataframe(pd.DataFrame({"Regular Market Price": [stock_data_info["Market Data"]["regularMarketPrice"]]}), hide_index=True, width=500)
    col2.dataframe(pd.DataFrame({"Regular Market Open": [stock_data_info["Market Data"]["regularMarketOpen"]]}), hide_index=True, width=500)
    col1, col2 = st.columns(2)
    col1.dataframe(pd.DataFrame({"Regular Market Day Low": [stock_data_info["Market Data"]["regularMarketDayLow"]]}), hide_index=True, width=500)
    col2.dataframe(pd.DataFrame({"Regular Market Day High": [stock_data_info["Market Data"]["regularMarketDayHigh"]]}), hide_index=True, width=500)
    col1, col2, col3 = st.columns(3)
    col1.dataframe(pd.DataFrame({"Fifty-Two Week Low": [stock_data_info["Market Data"]["fiftyTwoWeekLow"]]}), hide_index=True, width=300)
    col2.dataframe(pd.DataFrame({"Fifty-Two Week High": [stock_data_info["Market Data"]["fiftyTwoWeekHigh"]]}), hide_index=True, width=300)
    col3.dataframe(pd.DataFrame({"Fifty-Day Average": [stock_data_info["Market Data"]["fiftyDayAverage"]]}), hide_index=True, width=300)

    st.markdown("## **Volume and Shares**")
    col1, col2 = st.columns(2)
    col1.dataframe(pd.DataFrame({"Volume": [stock_data_info["Volume and Shares"]["volume"]]}), hide_index=True, width=500)
    col2.dataframe(pd.DataFrame({"Regular Market Volume": [stock_data_info["Volume and Shares"]["regularMarketVolume"]]}), hide_index=True, width=500)

    col1, col2, col3 = st.columns(3)
    col1.dataframe(pd.DataFrame({"Average Volume": [stock_data_info["Volume and Shares"]["averageVolume"]]}), hide_index=True, width=300)
    col2.dataframe(pd.DataFrame({"Average Volume (10 Days)": [stock_data_info["Volume and Shares"]["averageVolume10days"]]}), hide_index=True, width=300)
    col3.dataframe(pd.DataFrame({"Average Daily Volume (10 Day)": [stock_data_info["Volume and Shares"]["averageDailyVolume10Day"]]}), hide_index=True, width=300)

    col1.dataframe(pd.DataFrame({"Shares Outstanding": [stock_data_info["Volume and Shares"]["sharesOutstanding"]]}), hide_index=True, width=300)
    col2.dataframe(pd.DataFrame({"Implied Shares Outstanding": [stock_data_info["Volume and Shares"]["impliedSharesOutstanding"]]}), hide_index=True, width=300)
    col3.dataframe(pd.DataFrame({"Float Shares": [stock_data_info["Volume and Shares"]["floatShares"]]}), hide_index=True, width=300)

    st.markdown("## **Dividends and Yield**")
    col1, col2, col3 = st.columns(3)
    col1.dataframe(pd.DataFrame({"Dividend Rate": [stock_data_info["Dividends and Yield"]["dividendRate"]]}), hide_index=True, width=300)
    col2.dataframe(pd.DataFrame({"Dividend Yield": [stock_data_info["Dividends and Yield"]["dividendYield"]]}), hide_index=True, width=300)
    col3.dataframe(pd.DataFrame({"Payout Ratio": [stock_data_info["Dividends and Yield"]["payoutRatio"]]}), hide_index=True, width=300)

    st.markdown("## **Valuation and Ratios**")
    col1, col2 = st.columns(2)
    col1.dataframe(pd.DataFrame({"Market Cap": [stock_data_info["Valuation and Ratios"]["marketCap"]]}), hide_index=True, width=500)
    col2.dataframe(pd.DataFrame({"Enterprise Value": [stock_data_info["Valuation and Ratios"]["enterpriseValue"]]}), hide_index=True, width=500)
    col1.dataframe(pd.DataFrame({"Price to Book": [stock_data_info["Valuation and Ratios"]["priceToBook"]]}), hide_index=True, width=500)
    col2.dataframe(pd.DataFrame({"Debt to Equity": [stock_data_info["Valuation and Ratios"]["debtToEquity"]]}), hide_index=True, width=500)
    col1.dataframe(pd.DataFrame({"Gross Margins": [stock_data_info["Valuation and Ratios"]["grossMargins"]]}), hide_index=True, width=500)
    col2.dataframe(pd.DataFrame({"Profit Margins": [stock_data_info["Valuation and Ratios"]["profitMargins"]]}), hide_index=True, width=500)

    st.markdown("## **Financial Performance**")
    col1, col2 = st.columns(2)
    col1.dataframe(pd.DataFrame({"Total Revenue": [stock_data_info["Financial Performance"]["totalRevenue"]]}), hide_index=True, width=500)
    col2.dataframe(pd.DataFrame({"Revenue Per Share": [stock_data_info["Financial Performance"]["revenuePerShare"]]}), hide_index=True, width=500)

    col1, col2, col3 = st.columns(3)
    col1.dataframe(pd.DataFrame({"Total Cash": [stock_data_info["Financial Performance"]["totalCash"]]}), hide_index=True, width=300)
    col2.dataframe(pd.DataFrame({"Total Cash Per Share": [stock_data_info["Financial Performance"]["totalCashPerShare"]]}), hide_index=True, width=300)
    col3.dataframe(pd.DataFrame({"Total Debt": [stock_data_info["Financial Performance"]["totalDebt"]]}), hide_index=True, width=300)

    col1, col2 = st.columns(2)
    col1.dataframe(pd.DataFrame({"Earnings Growth": [stock_data_info["Financial Performance"]["earningsGrowth"]]}), hide_index=True, width=500)
    col2.dataframe(pd.DataFrame({"Revenue Growth": [stock_data_info["Financial Performance"]["revenueGrowth"]]}), hide_index=True, width=500)
    col1.dataframe(pd.DataFrame({"Return on Assets": [stock_data_info["Financial Performance"]["returnOnAssets"]]}), hide_index=True, width=500)
    col2.dataframe(pd.DataFrame({"Return on Equity": [stock_data_info["Financial Performance"]["returnOnEquity"]]}), hide_index=True, width=500)

    st.markdown("## **Cash Flow**")
    col1, col2 = st.columns(2)
    col1.dataframe(pd.DataFrame({"Free Cash Flow": [stock_data_info["Cash Flow"]["freeCashflow"]]}), hide_index=True, width=500)
    col2.dataframe(pd.DataFrame({"Operating Cash Flow": [stock_data_info["Cash Flow"]["operatingCashflow"]]}), hide_index=True, width=500)

    st.markdown("## **Analyst Targets**")
    col1, col2 = st.columns(2)
    col1.dataframe(pd.DataFrame({"Target High Price": [stock_data_info["Analyst Targets"]["targetHighPrice"]]}), hide_index=True, width=500)
    col2.dataframe(pd.DataFrame({"Target Low Price": [stock_data_info["Analyst Targets"]["targetLowPrice"]]}), hide_index=True, width=500)
    col1.dataframe(pd.DataFrame({"Target Mean Price": [stock_data_info["Analyst Targets"]["targetMeanPrice"]]}), hide_index=True, width=500)
    col2.dataframe(pd.DataFrame({"Target Median Price": [stock_data_info["Analyst Targets"]["targetMedianPrice"]]}), hide_index=True, width=500)

