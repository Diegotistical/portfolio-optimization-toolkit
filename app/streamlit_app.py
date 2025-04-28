# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from portfolio_optimizer.optimizer import PortfolioOptimizer
from src.backtest import run_backtest
from src.utils import calculate_performance_stats
from src.plotter import create_performance_plots

st.set_page_config(layout="wide")
st.title("Portfolio Optimization and Backtesting")

with st.sidebar:
    st.header("Parameters")
    tickers = st.text_input("Stock Tickers (comma-separated)", 
                           value="AAPL,MSFT,GOOG,AMZN,META,NVDA").upper().split(",")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))
    strategy = st.selectbox("Strategy", ["risk_parity", "mean_variance"])
    rebalance_freq = st.selectbox("Rebalance Frequency", ["QE", "ME", "YE"])
    transaction_cost = st.number_input("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.01) / 100
    risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100

if st.sidebar.button("Run Backtest"):
    with st.spinner("Running backtest..."):
        try:
            optimizer = PortfolioOptimizer(
                tickers=tickers,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                risk_free_rate=risk_free_rate
            )
            
            portfolio_eq, benchmark_eq, annual_turnover = run_backtest(
                optimizer=optimizer,
                strategy=strategy,
                rebalance_freq=rebalance_freq,
                transaction_cost=transaction_cost
            )
            
            stats = calculate_performance_stats(
                portfolio_eq=portfolio_eq,
                benchmark_eq=benchmark_eq,
                annual_turnover=annual_turnover,
                risk_free_rate=risk_free_rate
            )
            
            st.subheader("Performance Metrics")
            metrics = ['CAGR', 'Volatility', 'Sharpe', 'Max DD', 'Turnover']
            for metric in metrics:
                p = stats['Portfolio'][metric]
                b = stats['Benchmark'][metric]
                st.write(f"{metric:<12} | Portfolio: {p:.2%} | Benchmark: {b:.2%}")
            
            st.subheader("Performance Charts")
            fig = create_performance_plots(portfolio_eq, benchmark_eq)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.sidebar.markdown("""
**Instructions:**
1. Enter stock tickers separated by commas
2. Set your date range and strategy
3. Adjust transaction costs and risk-free rate
4. Click "Run Backtest" to see results
""")