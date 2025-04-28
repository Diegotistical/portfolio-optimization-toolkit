# src/plotter.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_performance_plots(portfolio_eq, benchmark_eq):
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