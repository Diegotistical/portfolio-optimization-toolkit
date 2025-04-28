# src/utils.py
import pandas as pd
import numpy as np

def calculate_performance_stats(portfolio_eq, benchmark_eq, annual_turnover, risk_free_rate):
    def safe_float(val):
        try:
            if isinstance(val, (pd.Series, pd.DataFrame)):
                return float(val.iloc[0])
            return float(val)
        except:
            return 0.0

    port_returns = portfolio_eq.pct_change().dropna()
    bench_returns = benchmark_eq.pct_change().dropna()

    stats = {
        'Portfolio': {
            'CAGR': safe_float((portfolio_eq.iloc[-1]/portfolio_eq.iloc[0]) ** (252/len(portfolio_eq)) - 1),
            'Volatility': safe_float(port_returns.std() * np.sqrt(252)),
            'Sharpe': safe_float((port_returns.mean() - risk_free_rate/252) * np.sqrt(252) / port_returns.std()),
            'Max DD': safe_float((portfolio_eq/portfolio_eq.cummax() - 1).min()),
            'Turnover': safe_float(annual_turnover)
        },
        'Benchmark': {
            'CAGR': safe_float((benchmark_eq.iloc[-1]/benchmark_eq.iloc[0]) ** (252/len(benchmark_eq)) - 1),
            'Volatility': safe_float(bench_returns.std() * np.sqrt(252)),
            'Sharpe': safe_float((bench_returns.mean() - risk_free_rate/252) * np.sqrt(252) / bench_returns.std()),
            'Max DD': safe_float((benchmark_eq/benchmark_eq.cummax() - 1).min()),
            'Turnover': 0.0
        }
    }
    return stats