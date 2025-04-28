import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from datetime import timedelta
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        self.tickers = tickers
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.risk_free_rate = risk_free_rate
        self.returns = self._get_data()
        self.previous_weights = None

    def _get_data(self):
        try:
            price_data = yf.download(
                self.tickers,
                start=self.start_date - timedelta(days=3*365),
                end=self.end_date,
                auto_adjust=True
            )['Close']
            price_data = price_data.dropna(axis=1, how='all').ffill().dropna()
            if len(price_data.columns) < 2:
                raise ValueError("Need at least 2 valid assets")
            returns = price_data.pct_change().dropna()
            if len(returns) < 63:
                raise ValueError("Insufficient data points (minimum 3 months)")
            return returns.astype(np.float64)
        except Exception as e:
            raise RuntimeError(f"Data error: {str(e)}")

    def _update_estimates(self, returns):
        cov_estimator = LedoitWolf().fit(returns)
        self.cov_matrix = pd.DataFrame(
            cov_estimator.covariance_,
            index=returns.columns,
            columns=returns.columns
        )
        self.expected_returns = returns.mean()

    def risk_parity(self):
        n = len(self.cov_matrix.columns)
        Sigma = self.cov_matrix.values
        initial_guess = np.ones(n)/n

        def _risk_contributions(w):
            port_var = w.T @ Sigma @ w
            marginal_risk = Sigma @ w
            return (w * marginal_risk) / port_var

        def objective(w):
            rc = _risk_contributions(w)
            target_rc = np.ones(n)/n
            return np.sum((rc - target_rc)**2)

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w}
        ]

        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1) for _ in range(n)],
            tol=1e-10
        )
        return result.x if result.success else np.ones(n)/n

    def mean_variance_optimization(self, target_return=None):
        n = len(self.cov_matrix.columns)
        mu = self.expected_returns.values
        Sigma = self.cov_matrix.values
        w = cp.Variable(n)
        portfolio_return = mu @ w
        portfolio_risk = cp.quad_form(w, Sigma)
        constraints = [
            cp.sum(w) == 1,
            w >= 0.01,
            w <= 0.5
        ]

        if target_return is not None:
            constraints.append(portfolio_return >= target_return)
            objective = cp.Minimize(portfolio_risk)
        else:
            excess_return = portfolio_return - self.risk_free_rate
            objective = cp.Maximize(excess_return / cp.sqrt(portfolio_risk))

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS)
            weights = w.value
            return np.clip(weights, 0, 0.5) if weights is not None else None
        except:
            return None

    def backtest(self, strategy='risk_parity', rebalance_freq='QE', 
                 benchmark='SPY', initial_capital=1e6, transaction_cost=0.001):
        valid_assets = self.returns.columns
        n_assets = len(valid_assets)
        benchmark_prices = yf.download(
            benchmark,
            start=self.start_date - timedelta(days=365*3),
            end=self.end_date,
            auto_adjust=True
        )['Close'].ffill()
        benchmark_rets = benchmark_prices.pct_change().fillna(0).astype(np.float64)
        current_value = initial_capital
        portfolio_values = [current_value]
        turnover_history = []
        weights = pd.DataFrame(index=self.returns.index, columns=valid_assets).astype(float)
        prev_weights = np.ones(n_assets)/n_assets
        valid_dates = self.returns.index[self.returns.index >= self.start_date]
        rebalance_dates = pd.date_range(
            start=valid_dates[0],
            end=valid_dates[-1],
            freq=rebalance_freq
        ).intersection(valid_dates)

        for i, date in enumerate(rebalance_dates):
            try:
                available_returns = self.returns.loc[:date - timedelta(days=1)]
                self._update_estimates(available_returns)
                if strategy == 'risk_parity':
                    new_weights = self.risk_parity()
                elif strategy == 'mean_variance':
                    new_weights = self.mean_variance_optimization()
                else:
                    new_weights = prev_weights

                if new_weights is None:
                    new_weights = prev_weights

                turnover = np.abs(new_weights - prev_weights).sum()
                turnover_history.append(turnover)
                cost = current_value * turnover * transaction_cost
                current_value -= cost
                prev_weights = new_weights
                end_date = rebalance_dates[i+1] if i < len(rebalance_dates)-1 else self.returns.index[-1]
                weights.loc[date:end_date] = new_weights
            except Exception as e:
                print(f"Rebalance error at {date}: {str(e)}")
                continue

        portfolio_rets = (self.returns * weights).sum(axis=1)
        portfolio_eq = initial_capital * (1 + portfolio_rets).cumprod()
        benchmark_eq = initial_capital * (1 + benchmark_rets).cumprod()
        annual_turnover = np.mean(turnover_history) * (252 / len(self.returns)) * len(rebalance_dates)
        return portfolio_eq.squeeze(), benchmark_eq.squeeze(), self._calculate_stats(portfolio_eq, benchmark_eq, annual_turnover)

    def _calculate_stats(self, portfolio, benchmark, annual_turnover):
        def safe_float(val):
            try:
                if isinstance(val, (pd.Series, pd.DataFrame)):
                    return float(val.iloc[0])
                return float(val)
            except:
                return 0.0

        port_returns = portfolio.pct_change().dropna()
        bench_returns = benchmark.pct_change().dropna()
        stats = {
            'Portfolio': {
                'CAGR': safe_float((portfolio.iloc[-1]/portfolio.iloc[0]) ** (252/len(portfolio)) - 1),
                'Volatility': safe_float(port_returns.std() * np.sqrt(252)),
                'Sharpe': safe_float((port_returns.mean() - self.risk_free_rate/252) * np.sqrt(252) / port_returns.std()),
                'Max DD': safe_float((portfolio/portfolio.cummax() - 1).min()),
                'Turnover': safe_float(annual_turnover)
            },
            'Benchmark': {
                'CAGR': safe_float((benchmark.iloc[-1]/benchmark.iloc[0]) ** (252/len(benchmark)) - 1),
                'Volatility': safe_float(bench_returns.std() * np.sqrt(252)),
                'Sharpe': safe_float((bench_returns.mean() - self.risk_free_rate/252) * np.sqrt(252) / bench_returns.std()),
                'Max DD': safe_float((benchmark/benchmark.cummax() - 1).min()),
                'Turnover': 0.0
            }
        }
        return stats

    def plot_results(self, portfolio_eq, benchmark_eq):
        initial_capital = portfolio_eq.iloc[0]
        cumulative_portfolio = (portfolio_eq / initial_capital - 1) * 100
        cumulative_benchmark = (benchmark_eq / initial_capital - 1) * 100
        dd_port = (portfolio_eq / portfolio_eq.cummax() - 1) * 100
        dd_bench = (benchmark_eq / benchmark_eq.cummax() - 1) * 100

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "Portfolio Value (Log Scale)",
                "Cumulative Returns (%)",
                "Drawdown Analysis (%)"
            )
        )

        fig.add_trace(go.Scatter(
            x=portfolio_eq.index, y=portfolio_eq,
            name="Strategy", line=dict(width=2.5, color="#636EFA"),
            hovertemplate="Date: %{x}<br>Value: %{y:$,.0f}"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=benchmark_eq.index, y=benchmark_eq,
            name="Benchmark", line=dict(width=2.5, dash="dash", color="#EF553B"),
            hovertemplate="Date: %{x}<br>Value: %{y:$,.0f}"
        ), row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=1)

        fig.add_trace(go.Scatter(
            x=cumulative_portfolio.index, y=cumulative_portfolio,
            name="Strategy Return", line=dict(width=2.5, color="#636EFA"),
            hovertemplate="Date: %{x}<br>Return: %{y:.1f}%"
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=cumulative_benchmark.index, y=cumulative_benchmark,
            name="Benchmark Return", line=dict(width=2.5, dash="dash", color="#EF553B"),
            hovertemplate="Date: %{x}<br>Return: %{y:.1f}%"
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=dd_port.index, y=dd_port,
            name="Strategy Drawdown", line=dict(width=2.5, color="#636EFA"),
            hovertemplate="Date: %{x}<br>Drawdown: %{y:.1f}%"
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=dd_bench.index, y=dd_bench,
            name="Benchmark Drawdown", line=dict(width=2.5, dash="dash", color="#EF553B"),
            hovertemplate="Date: %{x}<br>Drawdown: %{y:.1f}%"
        ), row=3, col=1)

        fig.update_layout(
            title="Portfolio Performance Analysis",
            height=900,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="seaborn",
            margin=dict(t=80, b=50, l=50, r=50)
        )
        return fig

# Streamlit UI
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
            
            portfolio_eq, benchmark_eq, stats = optimizer.backtest(
                strategy=strategy,
                rebalance_freq=rebalance_freq,
                transaction_cost=transaction_cost
            )
            
            st.subheader("Performance Metrics")
            metrics = ['CAGR', 'Volatility', 'Sharpe', 'Max DD', 'Turnover']
            for metric in metrics:
                p = stats['Portfolio'][metric]
                b = stats['Benchmark'][metric]
                st.write(f"{metric:<12} | Portfolio: {p:.2%} | Benchmark: {b:.2%}")
            
            st.subheader("Performance Charts")
            fig = optimizer.plot_results(portfolio_eq, benchmark_eq)
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