import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

def fetch_data(assets, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    # Fetch all columns of data for the given assets
    data = yf.download(assets, start=start_date, end=end_date)
    
    # Flatten the MultiIndex columns into a single level by combining asset ticker with column name (e.g., 'AAPL Close', 'AAPL Volume')
    data.columns = [f'{ticker} {col}' for ticker, col in data.columns]
    
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

def plot_stock_data(data):
    """Plot Open, Close, High, Low prices and Volume of assets."""
    plt.figure(figsize=(12, 6))

    # Identify relevant columns dynamically
    open_cols = [col for col in data.columns if 'Open' in col]
    high_cols = [col for col in data.columns if 'High' in col]
    low_cols = [col for col in data.columns if 'Low' in col]
    close_cols = [col for col in data.columns if 'Close' in col]
    volume_cols = [col for col in data.columns if 'Volume' in col]

    # Plot Open, High, Low, Close prices
    for col in open_cols:
        plt.plot(data.index, data[col], label=f'{col}', linestyle='dashed')
    for col in high_cols:
        plt.plot(data.index, data[col], label=f'{col}', linestyle='dotted')
    for col in low_cols:
        plt.plot(data.index, data[col], label=f'{col}', linestyle='dashdot')
    for col in close_cols:
        plt.plot(data.index, data[col], label=f'{col}', linewidth=2)

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Stock Prices (Open, High, Low, Close)")
    plt.legend(loc='upper left')

    # Create a second y-axis for Volume
    ax2 = plt.gca().twinx()
    for col in volume_cols:
        ax2.bar(data.index, data[col], alpha=0.3, color='gray', label=f'{col}')
    ax2.set_ylabel("Volume")

    # Show combined legends
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.show()
    
def plot_closing_price(data):
    """Plot closing prices of assets over time."""
    plt.figure(figsize=(12, 6))
    
    # Filter columns that contain 'Close'
    close_columns = [col for col in data.columns if 'Close' in col]
    
    for column in close_columns:
        plt.plot(data.index, data[column], label=column, linewidth=2)

    # Add title and labels
    plt.title("Closing Prices Over Time", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Closing Price", fontsize=12)

    # Improve readability
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    
    plt.show()

def plot_histogram(data, bins=50):
    """Plot histogram of asset prices for each column separately after normalization."""
    plt.figure(figsize=(12, 6))

    # Normalize each column separately for better visualization
    normalized_data = (data - data.min()) / (data.max() - data.min())

    # Plot histogram for each column separately
    for col in normalized_data.columns:
        plt.hist(normalized_data[col], bins=bins, alpha=0.5, label=col)

    # Add labels and title
    plt.title("Normalized Histogram of Asset Prices")
    plt.xlabel("Normalized Price (0 to 1 Scale)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.show()
    

def plot_daily_returns(data):
    """Plot daily percentage change in prices, excluding Volume columns."""
    
    # Exclude volume columns
    price_columns = [col for col in data.columns if "Volume" not in col]
    returns = data[price_columns].pct_change().dropna()

    # Plot the daily returns
    plt.figure(figsize=(12, 6))
    for column in returns.columns:
        plt.plot(returns.index, returns[column], label=column, alpha=0.7)

    plt.legend()
    plt.title("Daily Percentage Change in Prices")
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.show()
    return returns

def plot_volatility(data):
    """Plot rolling 30-day volatility (excluding Volume)."""
    
    price_columns = [col for col in data.columns if "Volume" not in col]
    returns = data[price_columns].pct_change().dropna()
    volatility = returns.rolling(window=30).std()

    plt.figure(figsize=(12, 6))
    for column in volatility.columns:
        plt.plot(volatility.index, volatility[column], label=column)
    
    plt.legend()
    plt.title("Rolling 30-day Volatility (Price Data)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

def detect_outliers(data):
    """Detect significant anomalies in daily returns and visualize them."""
    
    price_columns = [col for col in data.columns if "Volume" not in col]
    returns = data[price_columns].pct_change().dropna()
    
    # Compute Z-scores for outlier detection
    z_scores = (returns - returns.mean()) / returns.std()
    outliers = returns[(z_scores > 3) | (z_scores < -3)]


    plt.figure(figsize=(12, 6))
    sns.boxplot(data=returns, orient='h')
    plt.title("Boxplot of Daily Returns (Outliers Highlighted)")
    plt.show()
    
    return outliers

def analyze_volume_outliers(data):
    """Detect unusual spikes in trading volume."""
    
    volume_columns = [col for col in data.columns if "Volume" in col]
    volume_data = data[volume_columns]
    
    # Compute Z-scores for volume spikes
    z_scores = (volume_data - volume_data.mean()) / volume_data.std()
    volume_outliers = volume_data[(z_scores > 3) | (z_scores < -3)]

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=volume_data, orient='h')
    plt.title("Boxplot of Trading Volume (Outliers Highlighted)")
    plt.show()
    
    return volume_outliers

def plot_rolling_mean(data, window=30):
    """Plot rolling mean to identify trends (excluding Volume)."""
    
    price_columns = [col for col in data.columns if "Volume" not in col]
    plt.figure(figsize=(12, 6))

    for column in price_columns:
        plt.plot(data.index, data[column].rolling(window=window).mean(), label=f'{column} {window}-day MA')

    plt.legend()
    plt.title(f"Rolling {window}-day Mean of Tesla Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

def decompose_time_series(data, asset="Close TSLA"):
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