import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

def fetch_and_preprocess_data(ticker, start_date, end_date):
    """
    Fetch stock data from yfinance and preprocess it for time series forecasting.
    
    Parameters:
        ticker (str): The stock ticker symbol (e.g., 'TSLA').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        train (pd.Series): Training data (80% of the dataset).
        test (pd.Series): Testing data (20% of the dataset).
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Ensure data is not empty
    if data.empty:
        raise ValueError(f"No data fetched for {ticker}. Please check the ticker symbol and date range.")
    
    ts_data = data['Close'].dropna()  # Remove any NaN values
    train_size = int(len(ts_data) * 0.8)
    train, test = ts_data[:train_size], ts_data[train_size:]
    return train, test

def train_arima_model(train_data, test_data):
    """
    Train an ARIMA model using auto_arima and generate forecasts.
    
    Parameters:
        train_data (pd.Series or pd.DataFrame): Training data.
        test_data (pd.Series): Testing data.
    
    Returns:
        forecast (np.array): Forecasted values for the test data.
        model: Fitted ARIMA model.
    """
    # Ensure train_data is a pandas Series
    if isinstance(train_data, pd.DataFrame):
        train_data = train_data.iloc[:, 0]  # Use the first column if it's a DataFrame
    
    # Ensure there are no NaN values in the training data
    if train_data.isnull().any():
        print("Training data contains NaN values. Filling missing values with forward fill.")
        train_data = train_data.fillna(method='ffill')

    # Train the model using auto_arima to find the best (p, d, q) automatically
    model = auto_arima(train_data, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)

    # Fit the model to the training data
    model.fit(train_data)

    # Forecast for the length of the test data
    forecast = model.predict(n_periods=len(test_data))

    # If there are NaN values in the forecast, replace them with 0
    forecast = np.nan_to_num(forecast)

    return forecast, model
    
def train_sarima_model(train_data, test_data, seasonal_order=(1, 1, 1, 12)):
    """
    Train a SARIMA model and generate forecasts with confidence intervals.
    """
    model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    forecast = model_fit.get_forecast(steps=len(test_data))
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int()
    return forecast_values, model_fit


def train_lstm_model(train_data, test_data, n_steps=60, epochs=20, batch_size=32):
    """
    Train an LSTM model and generate forecasts.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1))
    
    X_train, y_train = [], []
    for i in range(n_steps, len(scaled_train)):
        X_train.append(scaled_train[i-n_steps:i, 0])
        y_train.append(scaled_train[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping])
    
    # Use the last n_steps from train_data to initialize predictions
    inputs = train_data[-n_steps:].values.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    
    # Prepare test data in the same way as training data
    X_test = []
    for i in range(n_steps, len(test_data) + n_steps):
        X_test.append(inputs[i-n_steps:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    
    return predicted_stock_price, model, scaler



def evaluate_model(test_data, forecast, model_name):
    """
    Evaluate the model's performance using MAE, RMSE, and MAPE.
    """
    # Ensure forecast is a pandas Series (convert if needed)
    if isinstance(forecast, np.ndarray):
        forecast = pd.Series(forecast, index=test_data.index)  # Match the index of the test data
    
    # Ensure test_data is a pandas Series
    if isinstance(test_data, pd.DataFrame):
        test_data = test_data.iloc[:, 0]  # Use the first column if it's a DataFrame
    
    # Ensure lengths match
    if len(forecast) != len(test_data):
        print(f"Warning: Length mismatch between {model_name} forecast and test data.")
        forecast = forecast[:len(test_data)]
    
    # Drop NaN values from both test data and forecast to avoid evaluation errors
    test_data = test_data.dropna()
    forecast = forecast.dropna()

    # Check for NaNs in the forecast and test data
    if np.any(np.isnan(test_data)) or np.any(np.isnan(forecast)):
        print(f"Warning: NaN values detected in {model_name} forecast or test data.")
        # Drop NaNs from both test_data and forecast
        test_data = test_data.dropna()
        forecast = forecast[~np.isnan(forecast)]

    # Check if forecast is empty
    if len(forecast) == 0:
        print(f"Error: Forecast is empty after removing NaN values. Check the model output.")
        return  # Exit the function as the forecast is empty

    # Ensure lengths match after handling NaNs
    if len(forecast) != len(test_data):
        print(f"Warning: Length mismatch after handling NaNs. Truncating forecast.")
        forecast = forecast[:len(test_data)]
    
    # Now you can calculate MAE, RMSE, and MAPE without NaN issues
    mae = mean_absolute_error(test_data, forecast)
    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
    
    print(f'{model_name} Model Evaluation:')
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAPE: {mape:.2f}%')
    
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data, label='Actual Prices', color='blue')
    plt.plot(test_data.index, forecast, label='Predicted Prices', color='red', linestyle='dashed')
    plt.title(f'{model_name} Model: Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
