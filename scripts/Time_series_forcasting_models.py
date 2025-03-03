import numpy as np
import pandas as pd
import yfinance as yf
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
    ts_data = data['Close']
    train_size = int(len(ts_data) * 0.8)
    train, test = ts_data[:train_size], ts_data[train_size:]
    return train, test

def train_arima_model(train_data, test_data):
    """
    Train an ARIMA model and generate forecasts.
    
    Parameters:
        train_data (pd.Series): Training data.
        test_data (pd.Series): Testing data.
    
    Returns:
        forecast (np.array): Forecasted values for the test data.
        model: Fitted ARIMA model.
    """
    model = auto_arima(train_data, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    model.fit(train_data)
    forecast = model.predict(n_periods=len(test_data), return_conf_int=False)
    return forecast, model

def train_sarima_model(train_data, test_data, seasonal_order=(1, 1, 1, 12)):
    """
    Train a SARIMA model and generate forecasts with confidence intervals.
    
    Parameters:
        train_data (pd.Series): Training data.
        test_data (pd.Series): Testing data.
        seasonal_order (tuple): Seasonal order (P, D, Q, S).
    
    Returns:
        forecast (np.array): Forecasted values for the test data.
        conf_int (np.array): Confidence intervals for the forecast.
        model: Fitted SARIMA model.
    """
    model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    forecast = model_fit.get_forecast(steps=len(test_data))
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int()
    return forecast_values, conf_int, model_fit

def train_lstm_model(train_data, test_data, n_steps=60, epochs=20, batch_size=32):
    """
    Train an LSTM model and generate forecasts.
    
    Parameters:
        train_data (pd.Series): Training data.
        test_data (pd.Series): Testing data.
        n_steps (int): Number of time steps to use for each sample.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
    
    Returns:
        forecast (np.array): Forecasted values for the test data.
        model: Trained LSTM model.
        scaler: Scaler object used for inverse transformation.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1))
    
    X_train, y_train = [], []
    for i in range(n_steps, len(scaled_train)):
        X_train.append(scaled_train[i-n_steps:i, 0])
        y_train.append(scaled_train[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[early_stopping])
    
    inputs = train_data[-n_steps:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    
    X_test = []
    for i in range(n_steps, len(inputs)):
        X_test.append(inputs[i-n_steps:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    
    return predicted_stock_price, model, scaler

def evaluate_model(test_data, forecast, model_name):
    """
    Evaluate the model's performance using MAE, RMSE, and MAPE.
    
    Parameters:
        test_data (pd.Series): Actual test data.
        forecast (np.array): Forecasted values.
        model_name (str): Name of the model (e.g., 'ARIMA', 'SARIMA', 'LSTM').
    """
    mae = mean_absolute_error(test_data, forecast)
    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
    
    print(f'{model_name} Model Evaluation:')
    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape}%')
    
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data, label='Actual Prices')
    plt.plot(test_data.index, forecast, label='Predicted Prices')
    plt.title(f'{model_name} Model: Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()