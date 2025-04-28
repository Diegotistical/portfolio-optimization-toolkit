A Python-based toolkit for portfolio optimization and backtesting, designed to help investors and analysts evaluate risk-adjusted returns using advanced strategies such as Risk Parity and Mean-Variance Optimization . The toolkit is powered by Streamlit for an interactive user interface and integrates seamlessly with financial data from Yahoo Finance .

Features
Advanced Optimization Strategies :
Risk Parity: Equal risk contribution across assets.
Mean-Variance Optimization: Maximize Sharpe ratio or target specific returns.
Backtesting Engine :
Simulates portfolio performance over historical data.
Supports customizable rebalancing frequencies (Quarterly, Monthly, Yearly).
Includes transaction cost modeling for realistic results.
Performance Metrics :
Compound Annual Growth Rate (CAGR).
Annualized Volatility.
Sharpe Ratio.
Maximum Drawdown.
Annual Turnover.
Interactive Visualizations :
Portfolio value over time (log scale).
Cumulative returns vs. benchmark.
Drawdown analysis.
Customizable Parameters :
Adjust transaction costs and risk-free rates.
Select stock tickers and date ranges.

Setup Instructions
Prerequisites
Python 3.8 or higher.
A modern web browser for the Streamlit interface.
Installation Steps
Clone the Repository :
bash

git clone https://github.com/Diegotistical/portfolio-optimization-toolkit.git
cd portfolio-optimization-toolkit
Set Up a Virtual Environment (optional but recommended):
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies :
bash


pip install -r requirements.txt
Run the Application :
bash
streamlit run app/streamlit_app.py
Open the provided local URL in your browser (e.g., http://localhost:8501).

Usage Guide
Input Parameters :
Enter stock tickers (comma-separated) in the sidebar (e.g., AAPL,MSFT,GOOG).
Set the start and end dates for the analysis.
Choose an optimization strategy (Risk Parity or Mean-Variance).
Configure rebalancing frequency, transaction costs, and risk-free rate.
Run Backtest :
Click the "Run Backtest" button to simulate portfolio performance.
View performance metrics and interactive charts.
Interpret Results :
Compare portfolio performance against a benchmark (default: SPY).
Analyze cumulative returns, drawdowns, and other key metrics.

Dependencies
The following Python libraries are required:

streamlit: For building the interactive web application.
numpy, pandas: For numerical computations and data manipulation.
yfinance: For downloading historical stock price data.
matplotlib, plotly: For visualizations.
cvxpy: For convex optimization problems.
scikit-learn: For covariance matrix estimation.
scipy: For numerical optimization.
Install all dependencies using:

bash
pip install -r requirements.txt
