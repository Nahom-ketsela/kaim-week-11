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

def plot_var_distribution(port_daily, confidence=0.95):
    """
    Plot a histogram of daily returns, and mark the VaR threshold line.
    """
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(port_daily, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Compute the VaR cutoff
    var_cutoff = np.percentile(port_daily, 100*(1 - confidence))
    
    # Add a vertical line for VaR
    plt.axvline(x=var_cutoff, color='red', linestyle='--', label=f"{confidence*100:.0f}% VaR = {var_cutoff:.2%}")
    plt.title("Distribution of Portfolio Daily Returns")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    

def plot_risk_return(annual_returns, annual_cov, rf=0.0, n_portfolios=10000):
    """
    Randomly generate many portfolios to show a risk-return scatter,
    then highlight the best Sharpe portfolio.
    
    annual_returns: Series with each asset's annualized return
    annual_cov: DataFrame or 2D array for annual covariance
    rf: risk-free rate
    n_portfolios: how many random portfolios to sample
    """
    np.random.seed(42)
    num_assets = len(annual_returns)
    
    results = []  # will hold (vol, ret, sharpe, weights)
    
    for _ in range(n_portfolios):
        w = np.random.random(num_assets)
        w /= np.sum(w)  # sum to 1
        
        port_ret = np.sum(w * annual_returns)
        port_vol = np.sqrt(w.T @ annual_cov.values @ w)
        sharpe = (port_ret - rf)/port_vol if port_vol != 0 else 0
        
        results.append((port_vol, port_ret, sharpe, w))
    
    # Convert to array for easier slicing
    results_array = np.array([ (r[0], r[1], r[2]) for r in results ])
    
    # Find portfolio with max Sharpe
    max_sharpe_idx = np.argmax(results_array[:,2])
    max_sharpe_port = results[max_sharpe_idx]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(results_array[:,0], results_array[:,1], c=results_array[:,2], cmap='viridis', alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(max_sharpe_port[0], max_sharpe_port[1], color='red', marker='*', s=300, label='Max Sharpe')
    plt.title("Portfolio Risk-Return Space")
    plt.xlabel("Volatility (Std Dev)")
    plt.ylabel("Return")
    plt.legend()
    plt.show()

def plot_portfolio_weights(weights, asset_labels):
    """
    Creates a pie chart for the final portfolio weights.
    weights: list or array of weights (e.g. [0.5, 0.3, 0.2])
    asset_labels: list of asset names matching the order of 'weights' (e.g. ['TSLA','BND','SPY'])
    """
    plt.figure(figsize=(6, 6))
    plt.pie(weights, labels=asset_labels, autopct='%1.1f%%', startangle=140)
    plt.title("Optimal Portfolio Allocation")
    plt.axis('equal')  # keep it a circle
    plt.show()

def plot_portfolio_cumulative(port_daily):
    """
    Plots the cumulative growth of the portfolio from an initial value of 1.0.
    port_daily: Series of the portfolio's daily returns (e.g., sum of asset returns * weights).
    """
    cumulative_growth = (1 + port_daily).cumprod()
    
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_growth.index, cumulative_growth, color='purple')
    plt.title("Portfolio Cumulative Return (Forecast Horizon)")
    plt.xlabel("Date")
    plt.ylabel("Growth (relative to initial 1.0)")
    plt.grid(True)
    plt.show()


def plot_asset_forecasts(df_forecast):
    """
    Plots the forecasted prices for each asset on a single line chart.
    df_forecast: DataFrame with columns [TSLA, BND, SPY], indexed by forecasted dates.
    """
    plt.figure(figsize=(10, 6))
    
    for col in df_forecast.columns:
        plt.plot(df_forecast.index, df_forecast[col], label=col)
    
    plt.title("Forecasted Daily Prices for Each Asset")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

