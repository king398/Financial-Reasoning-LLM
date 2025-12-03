"""Portfolio vs. S&P 500 benchmarking utilities.

This module provides a reproducible analytics workflow for comparing a user
portfolio against the SPY ETF benchmark. It builds daily return series from a
portfolio equity curve, aligns those returns with SPY, computes summary tables
for different periods, exports CSV reports, and generates diagnostic plots.

Primary entry point: `run_benchmark_analysis`.
"""

# Run `pip install -r requirements.txt` before using these utilities to ensure dependencies are available.

import dataclasses
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# Constants ---------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252
ROLLING_SHARPE_WINDOW = 63  # ~3 months
MIN_PERIOD_LENGTH = 60  # Minimum observations for annualised metrics


# Data containers ---------------------------------------------------------

@dataclasses.dataclass
class BenchmarkData:
    """Structured container for aligned portfolio/SPY returns."""

    portfolio_equity: pd.Series
    portfolio_returns: pd.Series
    spy_prices: pd.Series
    spy_returns: pd.Series
    active_returns: pd.Series
    risk_free_daily: float


# Loading & preprocessing -------------------------------------------------

def load_portfolio_equity(csv_path: str) -> pd.Series:
    """Load, sanitize, and return the equity curve as a Pandas Series.

    Usage:
        equity = load_portfolio_equity("path/to/portfolio_equity.csv")

    The CSV must contain a `date` column (parseable as YYYY-MM-DD) and an
    `equity` column representing end-of-day portfolio value. Dates are sorted,
    duplicate rows are removed, zero values are forward-filled, and any leading
    missing values are dropped.
    """

    df = pd.read_csv(csv_path, parse_dates=["date"])
    if "equity" not in df.columns:
        raise KeyError("CSV must include an 'equity' column.")

    df = df.sort_values("date").drop_duplicates("date")
    equity = df.set_index("date")["equity"].astype(float)

    # Guard against zero or missing values before computing returns.
    equity = equity.replace(0.0, np.nan).ffill()
    equity = equity.dropna()
    if equity.empty:
        raise ValueError("Portfolio equity series is empty after cleaning.")

    return equity


def _business_day_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Create a continuous business-day DateTimeIndex spanning the input window."""

    return pd.date_range(start=start, end=end, freq="B")


def _compute_daily_returns(series: pd.Series) -> pd.Series:
    """Convert a price-like series into simple daily percentage returns."""

    returns = series.pct_change().dropna()
    returns.name = "returns"
    return returns


def _fetch_spy_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Pull SPY prices via yfinance and return the adjusted close series."""

    spy = yf.download(
        "SPY",
        start=start.tz_localize(None),
        end=end.tz_localize(None) + pd.Timedelta(days=1),
        progress=False,
        auto_adjust=True,
    )
    if "Close" in spy.columns:
        prices = spy["Close"]
    elif "Adj Close" in spy.columns:
        prices = spy["Adj Close"]
    else:
        raise ValueError("Unexpected SPY data structure from yfinance.")

    # Newer versions of yfinance return MultiIndex columns even for single tickers.
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError("SPY price frame contains multiple columns unexpectedly.")
        prices = prices.squeeze(axis=1)

    prices.name = "SPY"
    prices = prices.dropna()
    if prices.empty:
        raise ValueError("SPY price series is empty.")

    return prices


def build_benchmark_data(
    equity: pd.Series,
    risk_free_rate_annual: float = 0.02,
) -> BenchmarkData:
    """Construct the aligned dataset required for downstream analytics.

    Usage:
        data = build_benchmark_data(equity_series, risk_free_rate_annual=0.03)

    The function forward-fills the equity curve onto a business-day calendar,
    fetches SPY prices for the same window, converts both to daily returns,
    aligns them on intersecting dates, derives active returns, and stores the
    daily risk-free rate implied by the annual input.
    """

    equity = equity.sort_index()
    calendar = _business_day_range(equity.index.min(), equity.index.max())
    equity = equity.reindex(calendar).ffill()

    portfolio_returns = _compute_daily_returns(equity)

    spy_prices = _fetch_spy_prices(calendar.min(), calendar.max())
    spy_prices = spy_prices.reindex(calendar).ffill()
    spy_returns = _compute_daily_returns(spy_prices)

    # Align on intersection to maintain synchronous observations.
    joined = pd.DataFrame(
        {
            "portfolio": portfolio_returns,
            "spy": spy_returns,
        }
    ).dropna()

    risk_free_daily = risk_free_rate_annual / TRADING_DAYS_PER_YEAR

    return BenchmarkData(
        portfolio_equity=equity.loc[joined.index.union([equity.index.min()])].ffill(),
        portfolio_returns=joined["portfolio"],
        spy_prices=spy_prices.loc[joined.index.union([spy_prices.index.min()])].ffill(),
        spy_returns=joined["spy"],
        active_returns=joined["portfolio"] - joined["spy"],
        risk_free_daily=risk_free_daily,
    )


# Metric calculations -----------------------------------------------------

def _max_drawdown(returns: pd.Series) -> Tuple[float, int]:
    """Compute the maximum drawdown and longest peak-to-recovery span."""

    wealth = (1.0 + returns).cumprod()
    running_max = wealth.cummax()
    drawdowns = wealth / running_max - 1.0

    tuw = 0
    current = 0
    for is_drawdown in drawdowns < 0:
        if is_drawdown:
            current += 1
            tuw = max(tuw, current)
        else:
            current = 0

    return drawdowns.min(), tuw


def _annualised_return(total_return: float, observations: int) -> float:
    """Convert a total compounded return into a CAGR estimate."""

    if observations < MIN_PERIOD_LENGTH or observations <= 0:
        return np.nan

    period_years = observations / TRADING_DAYS_PER_YEAR
    base = 1.0 + total_return
    if base <= 0:
        return np.nan

    return base ** (1.0 / period_years) - 1.0


def compute_metrics(
    returns: pd.Series,
    risk_free_daily: float,
) -> Dict[str, float]:
    """Calculate base performance, risk, and tail metrics for a return stream.

    Usage:
        stats = compute_metrics(portfolio_returns, risk_free_daily=0.0001)

    Expects daily simple returns indexed by trading date.
    """

    returns = returns.dropna()
    observations = len(returns)
    if observations == 0:
        return {metric: np.nan for metric in _metric_columns()}

    total_return = (1.0 + returns).prod() - 1.0
    volatility_daily = returns.std(ddof=0)
    volatility_annual = (
        volatility_daily * math.sqrt(TRADING_DAYS_PER_YEAR)
        if observations >= MIN_PERIOD_LENGTH
        else np.nan
    )

    excess = returns - risk_free_daily
    sharpe = np.nan
    if observations >= MIN_PERIOD_LENGTH and volatility_daily > 0:
        sharpe = excess.mean() / volatility_daily * math.sqrt(TRADING_DAYS_PER_YEAR)

    downside = excess.clip(upper=0)
    downside_std = downside.pow(2).mean() ** 0.5
    sortino = np.nan
    if observations >= MIN_PERIOD_LENGTH and downside_std > 0:
        sortino = excess.mean() / downside_std * math.sqrt(TRADING_DAYS_PER_YEAR)

    max_drawdown, time_under_water = _max_drawdown(returns)
    cagr = _annualised_return(total_return, observations)
    calmar = np.nan
    if observations >= MIN_PERIOD_LENGTH and max_drawdown < 0:
        calmar = cagr / abs(max_drawdown)

    hit_rate = (returns > 0).mean()
    var_95 = returns.quantile(0.05)
    tail_losses = returns[returns <= var_95]
    expected_shortfall = tail_losses.mean() if not tail_losses.empty else np.nan

    return {
        "observations": float(observations),
        "total_return": total_return,
        "cagr": cagr,
        "annualised_volatility": volatility_annual,
        "max_drawdown": max_drawdown,
        "time_under_water": float(time_under_water),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "hit_rate": hit_rate,
        "var_95": var_95,
        "expected_shortfall": expected_shortfall,
    }


def compare_vs_spy(
    portfolio_returns: pd.Series,
    spy_returns: pd.Series,
    risk_free_daily: float,
) -> Dict[str, float]:
    """Benchmark the portfolio against SPY and return active risk metrics.

    Usage:
        comps = compare_vs_spy(portfolio_returns, spy_returns, risk_free_daily)

    Returns annualised tracking error, information ratio, CAPM beta/alpha, and
    R-squared computed on the synchronized daily return series.
    """

    aligned = pd.DataFrame(
        {"portfolio": portfolio_returns, "spy": spy_returns}
    ).dropna()
    if aligned.empty:
        return {
            "tracking_error": np.nan,
            "information_ratio": np.nan,
            "beta": np.nan,
            "alpha": np.nan,
            "r_squared": np.nan,
        }

    active_returns = aligned["portfolio"] - aligned["spy"]
    observations = len(active_returns)

    active_vol_daily = active_returns.std(ddof=0)
    tracking_error = (
        active_vol_daily * math.sqrt(TRADING_DAYS_PER_YEAR)
        if observations >= MIN_PERIOD_LENGTH
        else np.nan
    )

    information_ratio = np.nan
    if observations >= MIN_PERIOD_LENGTH and tracking_error not in [0.0, np.nan]:
        active_mean = active_returns.mean()
        information_ratio = (
            active_mean * TRADING_DAYS_PER_YEAR
        ) / tracking_error

    spy_var = aligned["spy"].var(ddof=0)
    beta = np.nan
    alpha = np.nan
    r_squared = np.nan
    if observations >= 2 and spy_var > 0:
        covariance = np.cov(
            aligned["spy"], aligned["portfolio"], ddof=0
        )[0, 1]
        beta = covariance / spy_var

        avg_port = aligned["portfolio"].mean()
        avg_spy = aligned["spy"].mean()
        excess_port = avg_port - risk_free_daily
        excess_spy = avg_spy - risk_free_daily
        alpha_daily = excess_port - beta * excess_spy
        alpha = (
            (1.0 + alpha_daily) ** TRADING_DAYS_PER_YEAR - 1.0
            if alpha_daily > -1.0
            else np.nan
        )

        residuals = aligned["portfolio"] - (beta * aligned["spy"])
        ss_res = np.sum((residuals - residuals.mean()) ** 2)
        ss_tot = np.sum(
            (aligned["portfolio"] - aligned["portfolio"].mean()) ** 2
        )
        if ss_tot > 0:
            r_squared = 1 - ss_res / ss_tot

    return {
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "beta": beta,
        "alpha": alpha,
        "r_squared": r_squared,
    }


def _metric_columns() -> List[str]:
    """List the canonical metric keys produced by `compute_metrics`."""

    return [
        "observations",
        "total_return",
        "cagr",
        "annualised_volatility",
        "max_drawdown",
        "time_under_water",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "hit_rate",
        "var_95",
        "expected_shortfall",
    ]


def summarise_period(
    portfolio_returns: pd.Series,
    spy_returns: pd.Series,
    risk_free_daily: float,
) -> pd.DataFrame:
    """Produce a combined summary table for the portfolio, SPY, and comparatives.

    Usage:
        table = summarise_period(port_ret, spy_ret, risk_free_daily)
    """

    metrics_port = compute_metrics(portfolio_returns, risk_free_daily)
    metrics_spy = compute_metrics(spy_returns, risk_free_daily)
    comparative = compare_vs_spy(portfolio_returns, spy_returns, risk_free_daily)

    frames = [
        pd.Series(metrics_port, name="Portfolio"),
        pd.Series(metrics_spy, name="SPY"),
        pd.Series(comparative, name="Comparative"),
    ]

    summary = pd.DataFrame(frames).T
    return summary


def periodic_metrics(
    portfolio_returns: pd.Series,
    spy_returns: pd.Series,
    risk_free_daily: float,
    freq: str,
) -> Dict[str, pd.DataFrame]:
    """Generate per-period summary tables (calendar year or quarter).

    Usage:
        yearly = periodic_metrics(port_ret, spy_ret, rf_daily, freq="Y")
        quarterly = periodic_metrics(port_ret, spy_ret, rf_daily, freq="Q")
    """

    if freq not in {"Y", "Q"}:
        raise ValueError("Frequency must be 'Y' (yearly) or 'Q' (quarterly).")

    periods: Dict[str, pd.DataFrame] = {}
    for period, group in portfolio_returns.groupby(portfolio_returns.index.to_period(freq)):
        spy_group = spy_returns.loc[group.index]
        summary = summarise_period(group, spy_group, risk_free_daily)
        periods[str(period)] = summary

    return periods


# Plotting ---------------------------------------------------------------

def generate_plots(
    benchmark_data: BenchmarkData,
    output_dir: str,
) -> Dict[str, str]:
    """Create diagnostic PNG plots and return their file paths.

    Usage:
        plot_paths = generate_plots(benchmark_data, output_dir="reports/plots")
    """

    import matplotlib.pyplot as plt  # Imported lazily to avoid heavy import on load.

    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    returns = pd.DataFrame(
        {
            "Portfolio": benchmark_data.portfolio_returns,
            "SPY": benchmark_data.spy_returns,
        }
    )

    wealth = (1.0 + returns).cumprod()
    ax = wealth.plot(title="Cumulative Wealth")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, linestyle="--", alpha=0.5)
    wealth_path = os.path.join(output_dir, "cumulative_wealth.png")
    plt.tight_layout()
    plt.savefig(wealth_path)
    plt.close()
    paths["cumulative_wealth"] = wealth_path

    drawdowns = wealth / wealth.cummax() - 1.0
    ax = drawdowns.plot(title="Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.grid(True, linestyle="--", alpha=0.5)
    drawdown_path = os.path.join(output_dir, "drawdowns.png")
    plt.tight_layout()
    plt.savefig(drawdown_path)
    plt.close()
    paths["drawdowns"] = drawdown_path

    excess = returns - benchmark_data.risk_free_daily
    rolling_sharpe = (
        excess.rolling(window=ROLLING_SHARPE_WINDOW)
        .mean()
        .div(returns.rolling(window=ROLLING_SHARPE_WINDOW).std(ddof=0))
        * math.sqrt(TRADING_DAYS_PER_YEAR)
    )
    ax = rolling_sharpe.plot(title="Rolling 3-Month Sharpe Ratio")
    ax.set_ylabel("Sharpe")
    ax.grid(True, linestyle="--", alpha=0.5)
    sharpe_path = os.path.join(output_dir, "rolling_sharpe.png")
    plt.tight_layout()
    plt.savefig(sharpe_path)
    plt.close()
    paths["rolling_sharpe"] = sharpe_path

    return paths


# Output helpers ---------------------------------------------------------

def export_tables(
    overall: pd.DataFrame,
    yearly: Dict[str, pd.DataFrame],
    quarterly: Dict[str, pd.DataFrame],
    output_dir: str,
) -> Dict[str, str]:
    """Write summary tables to CSV files inside `output_dir`."""

    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    overall_path = os.path.join(output_dir, "overall_summary.csv")
    overall.to_csv(overall_path)
    paths["overall"] = overall_path

    yearly_path = os.path.join(output_dir, "yearly_summary.csv")
    _write_periodic_table(yearly, yearly_path)
    paths["yearly"] = yearly_path

    quarterly_path = os.path.join(output_dir, "quarterly_summary.csv")
    _write_periodic_table(quarterly, quarterly_path)
    paths["quarterly"] = quarterly_path

    return paths


def _write_periodic_table(
    tables: Dict[str, pd.DataFrame],
    output_path: str,
) -> None:
    """Helper to flatten a dict of DataFrames and export as a CSV."""

    rows: List[pd.DataFrame] = []
    for period, df in tables.items():
        tagged = df.copy()
        tagged.columns = pd.MultiIndex.from_product([[period], tagged.columns])
        rows.append(tagged)

    if not rows:
        pd.DataFrame().to_csv(output_path)
        return

    concatenated = pd.concat(rows, axis=1)
    concatenated.to_csv(output_path)


# Workflow ---------------------------------------------------------------

def run_benchmark_analysis(
    portfolio_csv_path: str,
    output_dir: str,
    risk_free_rate_annual: float = 0.02,
) -> Dict[str, Dict[str, str]]:
    """Execute the full benchmarking pipeline and return asset/plot paths.

    Usage:
        outputs = run_benchmark_analysis(
            portfolio_csv_path="data/portfolio_equity.csv",
            output_dir="reports/benchmark",
            risk_free_rate_annual=0.025,
        )

    Provide a CSV with `date` and `equity` columns. The function saves:
        - `overall_summary.csv`, `yearly_summary.csv`, `quarterly_summary.csv`
        - PNG plots for cumulative wealth, drawdowns, and rolling Sharpe
    """

    equity = load_portfolio_equity(portfolio_csv_path)
    benchmark_data = build_benchmark_data(
        equity,
        risk_free_rate_annual=risk_free_rate_annual,
    )

    overall_summary = summarise_period(
        benchmark_data.portfolio_returns,
        benchmark_data.spy_returns,
        benchmark_data.risk_free_daily,
    )

    yearly = periodic_metrics(
        benchmark_data.portfolio_returns,
        benchmark_data.spy_returns,
        benchmark_data.risk_free_daily,
        freq="Y",
    )

    quarterly = periodic_metrics(
        benchmark_data.portfolio_returns,
        benchmark_data.spy_returns,
        benchmark_data.risk_free_daily,
        freq="Q",
    )

    table_paths = export_tables(
        overall_summary,
        yearly,
        quarterly,
        output_dir=output_dir,
    )

    plot_paths = generate_plots(benchmark_data, output_dir=output_dir)

    return {
        "tables": table_paths,
        "plots": plot_paths,
    }


__all__ = [
    "BenchmarkData",
    "build_benchmark_data",
    "compare_vs_spy",
    "compute_metrics",
    "export_tables",
    "generate_plots",
    "load_portfolio_equity",
    "periodic_metrics",
    "run_benchmark_analysis",
    "summarise_period",
]
