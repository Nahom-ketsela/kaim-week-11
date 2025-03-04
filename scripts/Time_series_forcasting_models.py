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
    
    Parameters:
        train_data (pd.Series): Training data (must have a proper datetime index with frequency set).
        test_data (pd.Series): Testing data.
        seasonal_order (tuple): Seasonal order (P, D, Q, S).
    
    Returns:
        forecast (np.array): Forecasted values for the test data.
        model: Fitted SARIMA model.
    """
    # Ensure train_data index is a datetime index and set frequency if missing
    if not isinstance(train_data.index, pd.DatetimeIndex):
        train_data.index = pd.to_datetime(train_data.index)

    if train_data.index.freq is None:
        train_data = train_data.asfreq('D')  # Change 'D' to match your data's actual frequency (e.g., 'M' for monthly)

    model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    forecast = model_fit.get_forecast(steps=len(test_data))
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int()
    
    return forecast_values, conf_int, model_fit


def train_lstm_model(train_data, test_data, n_steps=60, epochs=20, batch_size=32):
    """
    Train an LSTM model and generate forecasts.
    """
    # Normalize training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1))
    
    # Prepare training sequences
    X_train, y_train = [], []
    for i in range(n_steps, len(scaled_train)):
        X_train.append(scaled_train[i - n_steps:i, 0])
        y_train.append(scaled_train[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Define LSTM model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model with early stopping
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping])


    # Concatenate the last `n_steps` from train_data with test_data
    full_series = pd.concat([train_data[-n_steps:], test_data])  
    inputs = scaler.transform(full_series.values.reshape(-1, 1))

    # Prepare test sequences
    X_test = []
    for i in range(n_steps, len(inputs)):
        X_test.append(inputs[i - n_steps:i, 0])
    
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predict test data
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


def forecast_lstm_future(
    model, 
    data_series,    
    scaler, 
    n_steps=60, 
    n_periods=126   
):
    """
    Forecast the next 'n_periods' time steps using an trained LSTM model.
    
    Parameters
    ----------
    model : Trained LSTM model (Keras).
    data_series : pd.Series
        The *full* historical data up to the end of 2025. 
        Must have a DatetimeIndex.
    scaler : e.g., MinMaxScaler or StandardScaler used in training.
    n_steps : int
        The window size the LSTM model expects (e.g., 60 days).
    n_periods : int
        Number of future steps to forecast (126 ~ 6 mo; 252 ~ 12 mo, if daily freq).

    Returns
    -------
    future_forecast : np.ndarray of shape (n_periods,)
    future_dates : pd.DatetimeIndex
    """

    # Ensure data is sorted by date
    data_series = data_series.sort_index()

    # Convert to numpy and scale
    full_data_arr = data_series.values.reshape(-1, 1)
    full_data_scaled = scaler.transform(full_data_arr)

    # Extract the final 'n_steps' from the dataset as the model input
    if len(full_data_scaled) < n_steps:
        raise ValueError("Not enough data to form a window of size n_steps.")
    last_window = full_data_scaled[-n_steps:]  # shape (n_steps, 1)

    # Iteratively predict n_periods steps ahead
    future_scaled_preds = []
    
    X_input = last_window.copy()  # shape (n_steps, 1)

    for _ in range(n_periods):
        # Reshape for LSTM: (1 batch, n_steps timesteps, 1 feature)
        X_input_reshaped = X_input.reshape(1, n_steps, 1)

        # Predict the next step
        pred_scaled = model.predict(X_input_reshaped)
        # pred_scaled is shape (1,1), take pred_scaled[0,0] => float
        next_val = pred_scaled[0, 0]

        # Append to future predictions
        future_scaled_preds.append(next_val)

        # Shift the window: drop the oldest, append the new prediction
        X_input = np.append(X_input[1:], [[next_val]], axis=0)

    # Inverse-transform the forecast to original scale
    future_forecast = scaler.inverse_transform(
        np.array(future_scaled_preds).reshape(-1, 1)
    ).flatten()

    # Create a future date range 
    last_date = data_series.index[-1]
    future_dates = pd.date_range(
        start=last_date, 
        periods=n_periods + 1, 
        freq='B'
    )[1:]  # skip the last known date => start from the day after

    # Plot the historical + forecast
    plt.figure(figsize=(10, 6))
    plt.plot(data_series.index, data_series, label='Historical Prices', color='blue')
    plt.plot(future_dates, future_forecast, label='Forecast (Next 6–12 Months)', color='orange')
    plt.title('LSTM Forecast')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    return future_forecast, future_dates
   

def interpret_results(future_forecast, future_dates):
    """
    Interpret the forecasted results to identify trends, volatility, and risks.
    
    Parameters:
        future_forecast (np.array): Forecasted values (length N).
        future_dates (pd.DatetimeIndex): Dates for the forecasted values (length N).
    """

    # Safety check
    if len(future_forecast) == 0 or len(future_dates) == 0:
        print("No forecast data to interpret.")
        return
    if len(future_forecast) != len(future_dates):
        print("Warning: future_forecast and future_dates are misaligned in length.")
    
    # Determine upward or downward trend based on first vs. last forecast
    first_val = future_forecast[0]
    last_val = future_forecast[-1]
    trend = 'upward' if last_val > first_val else 'downward'
    
    # Calculate approximate date range in days/months
    start_date = future_dates[0]
    end_date   = future_dates[-1]
    delta_days = (end_date - start_date).days
    approx_months = round(delta_days / 30.44) 
    
    # Percentage change from first to last forecast
    pct_change = (last_val - first_val) / first_val * 100 if first_val != 0 else 0
    
    # Measure volatility via standard deviation
    volatility_abs = np.std(future_forecast)
    mean_forecast  = np.mean(future_forecast)
    volatility_rel = (volatility_abs / mean_forecast * 100) if mean_forecast != 0 else 0

    # Print interpretation
    # Trend
    print(f"Forecast Period: {start_date.date()} to {end_date.date()} (~{approx_months} months)")
    print(f"Trend: The forecast suggests a {trend} movement over this period.")
    print(f"Price change: from {first_val:.2f} to {last_val:.2f} ({pct_change:+.2f}%).")

    # Volatility
    print(f"Volatility (std): {volatility_abs:.2f}")
    print(f"Relative Volatility: {volatility_rel:.2f}% of average forecast value.\n")
    
    # Market opportunity/risk commentary
    if trend == 'upward':
        print("Potential Market Opportunity: The expected price increase may present buying opportunities.")
    else:
        print("Potential Market Risk: The expected price decline suggests caution in investment (or shorting opportunities).")
    
    # Additional risk commentary if volatility is high
    if volatility_rel > 10:
        print("Note: Volatility appears relatively high (std > 10% of mean). "
              "This implies greater uncertainty in the forecast.")
    else:
        print("Volatility appears moderate, implying relatively stable expectations under the model.")