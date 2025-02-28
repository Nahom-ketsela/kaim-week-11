import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

def fetch_data(assets, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    data = yf.download(assets, start=start_date, end=end_date)["Adj Close"]
    data.columns = assets  # Rename columns for clarity
    return data

def check_missing_values(data):
    """Check for missing values in the dataset."""
    missing = data.isnull().sum()
    print("Missing values per asset:")
    print(missing)
    return missing

def preprocess_data(data):
    """Handle missing values and normalize data."""
    data.fillna(method='ffill', inplace=True)
    data.index = pd.to_datetime(data.index)
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    return data, scaled_data

def plot_closing_prices(data):
    """Plot closing prices of assets."""
    plt.figure(figsize=(12,6))
    for column in data.columns:
        plt.plot(data.index, data[column], label=column)
    plt.legend()
    plt.title("Closing Prices of Assets")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()

def plot_histogram(data):
    """Plot histogram of asset prices."""
    data.plot(kind='hist', bins=50, figsize=(12,6), alpha=0.7)
    plt.title("Histogram of Asset Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.legend(data.columns)
    plt.show()

def plot_daily_returns(data):
    """Plot daily percentage change in prices."""
    returns = data.pct_change().dropna()
    plt.figure(figsize=(12,6))
    for column in returns.columns:
        plt.plot(returns.index, returns[column], label=column)
    plt.legend()
    plt.title("Daily Percentage Change in Prices")
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.show()
    return returns

def plot_volatility(returns):
    """Plot rolling 30-day volatility."""
    volatility = returns.rolling(window=30).std()
    plt.figure(figsize=(12,6))
    for column in volatility.columns:
        plt.plot(volatility.index, volatility[column], label=column)
    plt.legend()
    plt.title("Rolling 30-day Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.show()

def detect_outliers(returns):
    """Detect significant anomalies in daily returns and visualize them."""
    z_scores = (returns - returns.mean()) / returns.std()
    outliers = returns[(z_scores > 3) | (z_scores < -3)]
    print("Outliers detected:")
    print(outliers)
    plt.figure(figsize=(12,6))
    sns.boxplot(data=returns, orient='h')
    plt.title("Boxplot of Daily Returns (Outliers Highlighted)")
    plt.show()
    return outliers

def plot_rolling_mean(data, window=30):
    """Plot rolling mean to identify trends."""
    plt.figure(figsize=(12,6))
    for column in data.columns:
        plt.plot(data.index, data[column].rolling(window=window).mean(), label=f'{column} {window}-day MA')
    plt.legend()
    plt.title(f"Rolling {window}-day Mean of Asset Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()

def decompose_time_series(data, asset):
    """Decompose time series into trend, seasonal, and residual components."""
    decomposition = seasonal_decompose(data[asset], model='additive', period=365)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    decomposition.observed.plot(ax=ax1, title="Observed")
    decomposition.trend.plot(ax=ax2, title="Trend")
    decomposition.seasonal.plot(ax=ax3, title="Seasonal")
    decomposition.resid.plot(ax=ax4, title="Residual")
    plt.tight_layout()
    plt.show()
    return decomposition
