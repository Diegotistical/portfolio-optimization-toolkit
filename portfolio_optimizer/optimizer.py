# portfolio_optimizer/optimizer.py
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
import cvxpy as cp
from src.data_loader import load_and_prepare_data

class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.02):
        self.tickers = tickers
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.risk_free_rate = risk_free_rate
        self.returns = load_and_prepare_data(tickers, start_date, end_date)
        self.previous_weights = None

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