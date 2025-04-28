# src/backtest.py
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta

def run_backtest(optimizer, strategy='risk_parity', rebalance_freq='QE',
                 benchmark='SPY', initial_capital=1e6, transaction_cost=0.001):
    
    valid_assets = optimizer.returns.columns
    n_assets = len(valid_assets)
    
    benchmark_prices = yf.download(
        benchmark,
        start=optimizer.start_date - timedelta(days=365*3),
        end=optimizer.end_date,
        auto_adjust=True
    )['Close'].ffill()
    
    benchmark_rets = benchmark_prices.pct_change().fillna(0).astype(np.float64)
    
    current_value = initial_capital
    portfolio_values = [current_value]
    turnover_history = []
    weights = pd.DataFrame(index=optimizer.returns.index, columns=valid_assets).astype(float)
    prev_weights = np.ones(n_assets)/n_assets
    
    valid_dates = optimizer.returns.index[optimizer.returns.index >= optimizer.start_date]
    rebalance_dates = pd.date_range(
        start=valid_dates[0],
        end=valid_dates[-1],
        freq=rebalance_freq
    ).intersection(valid_dates)
    
    for i, date in enumerate(rebalance_dates):
        try:
            available_returns = optimizer.returns.loc[:date - timedelta(days=1)]
            optimizer._update_estimates(available_returns)
            
            if strategy == 'risk_parity':
                new_weights = optimizer.risk_parity()
            elif strategy == 'mean_variance':
                new_weights = optimizer.mean_variance_optimization()
            else:
                new_weights = prev_weights
                
            if new_weights is None:
                new_weights = prev_weights
                
            turnover = np.abs(new_weights - prev_weights).sum()
            turnover_history.append(turnover)
            cost = current_value * turnover * transaction_cost
            current_value -= cost
            prev_weights = new_weights
            
            end_date = rebalance_dates[i+1] if i < len(rebalance_dates)-1 else optimizer.returns.index[-1]
            weights.loc[date:end_date] = new_weights
            
        except Exception as e:
            print(f"Rebalance error at {date}: {str(e)}")
            continue
            
    portfolio_rets = (optimizer.returns * weights).sum(axis=1)
    portfolio_eq = initial_capital * (1 + portfolio_rets).cumprod()
    benchmark_eq = initial_capital * (1 + benchmark_rets).cumprod()
    annual_turnover = np.mean(turnover_history) * (252 / len(optimizer.returns)) * len(rebalance_dates)
    
    return portfolio_eq.squeeze(), benchmark_eq.squeeze(), annual_turnover