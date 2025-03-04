import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def naive_forecast(asset_series, forecast_horizon=126):
    """
    Naive forecast: compute average daily return over the last 60 days
    and project it forward for 'forecast_horizon' days.
    Returns a NumPy array of future prices.
    """
    asset_series = asset_series.dropna().sort_index()
    
    # We'll use the last 60 days to measure average daily return
    window_days = 60
    recent_data = asset_series.iloc[-window_days:]
    daily_returns = recent_data.pct_change().dropna()
    
    avg_daily_ret = daily_returns.mean()
    
    last_price = asset_series.iloc[-1]
    
    # Generate forecast by compounding the average daily return
    future_prices = np.zeros(forecast_horizon)
    current_price = last_price
    
    for i in range(forecast_horizon):
        current_price *= (1 + avg_daily_ret)
        future_prices[i] = current_price
    
    return future_prices

def portfolio_performance(weights, annual_returns, annual_cov, rf=0.0):
    """
    Calculate the portfolio's annual return and volatility given weights, 
    as well as the Sharpe ratio.
    """
    w = np.array(weights)
    port_return = np.sum(w * annual_returns)
    port_vol = np.sqrt(w.T @ annual_cov.values @ w)
    # Sharpe ratio
    sharpe = (port_return - rf) / port_vol if port_vol != 0 else 0.0
    return port_return, port_vol, sharpe

def negative_sharpe_ratio(weights, annual_returns, annual_cov, rf=0.0):
    """
    Helper function: returns the negative Sharpe so we can 'minimize' it.
    """
    _, _, sharpe = portfolio_performance(weights, annual_returns, annual_cov, rf)
    return -sharpe

def weight_sum_constraint(weights):
    """
    Constraint for sum of weights = 1.0
    """
    return np.sum(weights) - 1.0

def optimize_portfolio(df_prices, rf=0.0):
    """
    Given a DataFrame of forecasted daily prices for multiple assets, 
    compute daily returns, annualize, and find the optimal weights 
    that maximize the Sharpe ratio (Markowitz optimization).
    
    df_prices: DataFrame with columns for each asset, each column is daily forecasted prices.
    rf: risk-free rate (default 0.0).
    
    Returns
    -------
    optimal_weights : np.array
    stats : dict with 'return', 'volatility', 'sharpe'
    """
    #  Daily returns
    returns_df = df_prices.pct_change().dropna()
    
    # Annualize
    daily_means = returns_df.mean()
    annual_returns = daily_means * 252
    annual_cov = returns_df.cov() * 252
    
    # Setup optimization
    num_assets = len(df_prices.columns)
    init_guess = [1.0/num_assets]*num_assets
    bounds = [(0,1)]*num_assets
    constraints = [{'type':'eq', 'fun':weight_sum_constraint}]
    
    # Minimize negative Sharpe
    opt_result = minimize(
        negative_sharpe_ratio,
        x0=init_guess,
        args=(annual_returns, annual_cov, rf),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    optimal_weights = opt_result.x
    
    # 5) Evaluate final performance
    final_return, final_vol, final_sharpe = portfolio_performance(
        optimal_weights, annual_returns, annual_cov, rf
    )
    
    stats = {
        'return': final_return,
        'volatility': final_vol,
        'sharpe': final_sharpe
    }
    return optimal_weights, stats

def compute_var(daily_returns, confidence=0.95):
    """
    Compute the (1 - confidence) percentile as Value at Risk (historical).
    E.g., 95% VaR means we look at the 5% worst daily returns.
    """
    var_cutoff = np.percentile(daily_returns, 100*(1-confidence))
    return -var_cutoff  # negative sign => 'loss' as a positive number