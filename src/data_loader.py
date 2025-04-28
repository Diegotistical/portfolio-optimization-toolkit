# src/data_loader.py
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta

def load_and_prepare_data(tickers, start_date, end_date):
    """
    Load and preprocess price data for portfolio optimization
    
    Args:
        tickers (list): List of stock tickers
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        pd.DataFrame: Cleaned daily returns data
    """
    try:
        # Download historical price data with 3-year buffer for covariance estimation
        price_data = yf.download(
            tickers,
            start=pd.to_datetime(start_date) - timedelta(days=3*365),
            end=pd.to_datetime(end_date),
            auto_adjust=True
        )['Close']
        
        # Data cleaning and validation
        price_data = price_data.dropna(axis=1, how='all').ffill().dropna()
        if len(price_data.columns) < 2:
            raise ValueError("Need at least 2 valid assets")
            
        returns = price_data.pct_change().dropna()
        if len(returns) < 63:
            raise ValueError("Insufficient data points (minimum 3 months)")
            
        return returns.astype(np.float64)
    
    except Exception as e:
        raise RuntimeError(f"Data loading error: {str(e)}")