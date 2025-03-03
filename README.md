

# GMF Investments - Portfolio Optimization & Time Series Forecasting  

## Overview  

**GMF Investments** is a financial advisory project that applies advanced **time series forecasting** and **portfolio optimization** techniques to enhance investment decision-making. By leveraging real-time financial data, the project aims to predict market trends, optimize asset allocation, and minimize portfolio risks while maximizing returns.  

## Project Goals  

- **Forecast Market Trends**: Utilize historical stock price data to predict future movements.  
- **Optimize Asset Allocation**: Adjust portfolio weightings to maximize returns while managing risk.  
- **Enhance Risk Management**: Analyze volatility, Value at Risk (VaR), and Sharpe ratios.  
- **Data-Driven Decision Making**: Use financial data from **YFinance** to make informed investment recommendations.  

## Datasets Used  

Historical financial data (Jan 2015 – Jan 2025) for three key assets:  

| Asset | Description | Role in Portfolio |  
|---|---|---|  
| **Tesla (TSLA)** | High-growth stock, volatile | High returns, high risk |  
| **Vanguard Total Bond Market ETF (BND)** | Bond ETF, stable | Low risk, stability |  
| **S&P 500 ETF (SPY)** | Broad market exposure | Moderate risk, diversification |  

Each dataset includes: **Date, Open, High, Low, Close, Adj Close, and Volume**.  

## Project Structure  

### **1. Data Preprocessing & Exploration**  
- Extracted financial data using **YFinance**.  
- Cleaned data: Handled missing values, normalized features, and ensured consistency.  
- Conducted **Exploratory Data Analysis (EDA)**:  
  - Price trend visualization.  
  - Daily percentage change and volatility analysis.  
  - Rolling statistics for trend detection.  
  - Seasonality and decomposition.  

### **2. Time Series Forecasting**  
Developed predictive models to forecast **Tesla’s stock price** using:  
- **ARIMA**: Best for short-term trends.  
- **SARIMA**: Captures seasonal patterns.  
- **LSTM**: Deep learning model for complex patterns.  

#### **Model Evaluation Metrics**  
- **MAE (Mean Absolute Error)**  
- **RMSE (Root Mean Squared Error)**  
- **MAPE (Mean Absolute Percentage Error)**  

### **3. Forecast Future Market Trends**  
- Generated **6-12 month** future price predictions.  
- Incorporated **confidence intervals** to measure forecast uncertainty.  
- Identified potential risks and investment opportunities based on trends.  

### **4. Portfolio Optimization**  
- Created a diversified portfolio using **TSLA, BND, and SPY**.  
- Calculated **expected returns, volatility, and risk correlations**.  
- **Optimized portfolio weights** to maximize the **Sharpe Ratio** (risk-adjusted return).  
- Conducted **Value at Risk (VaR)** analysis to estimate potential losses.  

### **5. Visualization & Insights**  
- **Stock trend graphs** for historical and forecasted prices.  
- **Volatility and risk-return scatter plots**.  
- **Cumulative return charts** to compare different portfolio allocations.  

## Technologies & Tools  

- **Data Extraction**: `YFinance`  
- **Data Manipulation**: `Pandas`, `NumPy`  
- **Visualization**: `Matplotlib`, `Seaborn`  
- **Time Series Modeling**: `Statsmodels`, `pmdarima`, `TensorFlow/Keras`  
- **Portfolio Optimization**: `SciPy`  

## How to Run  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Nahom-ketsela/kaim-week-11.git  
   
   ```  

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt  
   ```  

3. **Run the Jupyter Notebook**  
   ```bash
   jupyter notebook  
   ```  

4. **Execute Forecasting & Optimization Scripts**  

## Expected Outcomes  

- **Accurate stock price predictions** for better market timing.  
- **Optimized portfolio allocation** for risk-adjusted returns.  
- **Deeper insights into market trends** and asset volatility.  


