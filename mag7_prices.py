import numpy as np
import pandas as pd
import yfinance as yf
import talib
from backtesting import Backtest, Strategy
from scipy.stats import spearmanr
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
import plotly.graph_objects as go

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.4f}".format)
np.random.seed(42)

#1. Global Research Configuration

Tickers = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
]

# Sample window
End_date = pd.Timestamp.today().normalize()
Start_date = End_date - pd.DateOffset(years=5)

# Signal parameters
mom_lookback = 252
mom_skip = 21

# Cross Sectional processing
winsor_lo = 0.01
winsor_hi = 0.99

# Portfolio Construction
max_gross_leverage = 1.0

#2. Project paths and structure

project_root = Path(__file__).resolve().parents[1]
print("PROJECT ROOT:", project_root.resolve())

Data_raw = project_root / "data" / "raw"
Data_processed = project_root / "data" / "processed"
Output_figures = project_root / "output" / "figures"
Output_metrics = project_root / "output" / "metrics"

# Create folders if they dont exist 
for path in [
    Data_raw,
    Data_processed,
    Output_figures,
    Output_metrics,
]:
    path.mkdir(parents=True, exist_ok=True)

#3. Print funcs (sanity check)
print("Configuration")
print("-------------")
print(f"Tickers: {Tickers}")
print(f"Sample window: {Start_date.date()} -> {End_date.date()}")
print(f"Momentum lookback: {mom_lookback}, skip: {mom_skip}")
print(f"Winsorization: {winsor_lo}-{winsor_hi}")

#4. Data download (Yahoo Finance)
print("\nDownloading price data from Yahoo Finance...")

raw_data = yf.download(
    tickers=Tickers,
    start=Start_date,
    end=End_date,
    auto_adjust=False,
    progress=True
)

# Extract Adjusted Close prices
prices = raw_data["Adj Close"].copy()

# Ensure datetime index
prices.index = pd.to_datetime(prices.index)

# Sort index
prices = prices.sort_index()

# Basic sanity checks
print("\nData checks")
print("-------------")
print("Shape: ", prices.shape)
print("Data range: ", prices.index.min().date(), "->", prices.index.max().date())
print("\nMissing values per ticker: ")
print(prices.isna().sum())

# Save raw data

raw_file = Data_raw / "mag7_prices.csv"
prices.to_csv(raw_file)

print(f"\nRaw price data saved to: {raw_file.resolve()}")

#5. Returns Construction and Processed Data

# Load raw prices 
prices = pd.read_csv(
    Data_raw / "mag7_prices.csv", 
    index_col=0, 
    parse_dates=True
).sort_index()

# Compute daily returns 
returns = np.log(prices).diff()

# Sanity checks on returns
print("\nReturns Checks")
print("----------------")
print("Shape: ", returns.shape)
print("NaN count:", returns.isna().sum().sum())
print("\nSummary stats: ")
print(returns.describe().T[["mean", "std", "min", "max"]])

# Drop first row (log diff artifact)
returns = returns.dropna()

# Save processed data (FORCED schema)
processed_prices_file = Data_processed / "mag7_prices.feather"
processed_returns_file = Data_processed / "mag7_returns.feather"

prices_reset = prices.reset_index()
prices_reset.columns = ["date"] + list(prices_reset.columns[1:])
prices_reset.to_feather(processed_prices_file)

returns_reset = returns.reset_index()
returns_reset.columns = ["date"] + list(returns_reset.columns[1:])
returns_reset.to_feather(processed_returns_file)

print("\nProcessed data saved (forced date column):")
print(processed_prices_file.resolve())
print(processed_returns_file.resolve())


#6. Momentum Signal Construction
# Load processed prices 
prices = pd.read_feather(
    Data_processed / "mag7_prices.feather"
).set_index("date").sort_index() 

# Compute momentum signal with TA-lib
momentum = pd.DataFrame(index=prices.index, columns=prices.columns)

for ticker in prices.columns:
    price_series = prices[ticker].values.astype(float)

    #Full 12-month momentum
    mom_long = talib.ROC(price_series, timeperiod=mom_lookback)

    #Short-term (1-month) 
    mom_short = talib.ROC(price_series, timeperiod=mom_skip)

    # Net momentum = long-term minus short-term
    momentum[ticker] = mom_long - mom_short

# Net momentum = skip most recent month
momentum[ticker] = mom_long - mom_short

momentum = momentum.dropna()

# Save raw momentum signal 
momentum_file = Data_processed / "mag7_momentum_raw.feather"
momentum.reset_index().to_feather(momentum_file)

print("\nRaw momentum signal saved to: ")
print(momentum_file.resolve())

# Signal sanity checks
print(momentum.tail())
print(momentum.isna().sum())


#7. Load raw momentum signal (defensive)

momentum = pd.read_feather(
    Data_processed / "mag7_momentum_raw.feather"
).set_index("date").sort_index()

# Define cross-sectional helper functions

def winsorize_series(x, lo=winsor_lo, hi=winsor_hi):
    lower = x.quantile(lo)
    upper = x.quantile(hi)
    return x.clip(lower, upper)

def zscore_series(x):
    mu = x.mean()
    sigma = x.std(ddof=0)
    if sigma == 0:
        return x * 0.0
    return (x - mu) / sigma

momentum_winsor = momentum.apply(
    winsorize_series,
    axis=1
)

# Apply cross-sectional z-scoring 
momentum_z = momentum_winsor.apply(
    zscore_series,
    axis=1
)

# Sanity checks 
print("\nCross-sectional checks")
print("------------------------")
print("Row means (last 5 day): ")
print(momentum_z.tail().mean(axis=1))

print("\nRow std devs (last 5 days): ")
print(momentum_z.tail().std(axis=1))

#Inspect a snapshot
print("\nSample standardized scores: ")
print(momentum_z.tail())

# Save standardized alpha signal 
alpha_file = Data_processed / "mag7_momentum_zscore.feather"

momentum_z.reset_index().to_feather(alpha_file)

print("\nStandardized alpha saved to: ")
print(alpha_file.resolve())

#8. Load standardized alpha signal (defensive)

alpha = pd.read_feather(
    Data_processed / "mag7_momentum_zscore.feather"
).set_index("date").sort_index()

# Alpha sanity check 
print("\nAlpha checks")
print("--------------")
print(alpha.tail())
print("\nRow mean (should be ~ 0): ")
print(alpha.tail().mean(axis=1))

# Map alpha --> raw weights
def alpha_to_weights(x):
    # Rank within the day
    ranks = x.rank(method="first")

    # Center around zero
    centered = ranks - ranks.mean()

    return centered

raw_weights = alpha.apply(alpha_to_weights, axis=1)

# Enforce dollar neutrality and leverage constraint
weights = raw_weights.div(
    raw_weights.abs().sum(axis=1),
    axis=0
) * max_gross_leverage

# Final weight checks 
print("\nWeight checks")
print("---------------")

print("Row sum (should be ~0): ")
print(weights.tail().sum(axis=1))

print("\nGross leverage (should be 1.0): ")
print(weights.tail().abs().sum(axis=1))

# Save portfolio weights
weights_file = Data_processed / "mag7_portfolio_weights.feather"

weights.reset_index().to_feather(weights_file)

print("\nPortfolio weights saved to: ")
print(weights_file.resolve())


# 8. Porfolio PnL

# Load returns (defensive)
returns = pd.read_feather(
    Data_processed / "mag7_returns.feather"
).set_index("date").sort_index()

# Load portfolio weights
weights = pd.read_feather(
    Data_processed / "mag7_portfolio_weights.feather"
).set_index("date").sort_index()

# Sanity check
print(returns.shape)
print(weights.shape)


# Align weights and returns -- shift weights forward to avoid look-ahead bias
weights_aligned = weights.shift(1)

# Align dates
common_dates = weights_aligned.index.intersection(returns.index)

weights_aligned = weights_aligned.loc[common_dates]
returns_aligned = returns.loc[common_dates]

# Drop first row caused by shift
weights_aligned = weights_aligned.dropna()
returns_aligned = returns_aligned.loc[weights_aligned.index]

# Compute daily portfolio PnL
portfolio_returns = (weights_aligned * returns_aligned).sum(axis=1)

# Cumulative returns (log-return consistent)
cumulative_returns = portfolio_returns.cumsum()

# Minimal sanity checks 
print("\nPortfolio PnL checks")
print("----------------------")
print("Daily returns summary: ")
print(portfolio_returns.describe())

print("\nLast 5 cumulative returns: ")
print(cumulative_returns.tail())

# Save PnL output
pnl_file = Output_metrics / "portfolio_pnl.feather"

pd.DataFrame({
    "portfolio_returns": portfolio_returns,
    "cumulative_returns": cumulative_returns
}).reset_index().to_feather(pnl_file)

print("\nPortfolio PnL saved to: ")
print(pnl_file.resolve())


# 9. Information Coefficient (IC)

# Load standardized alpha 
alpha = pd.read_feather(
    Data_processed / "mag7_momentum_zscore.feather"
).set_index("date").sort_index()

# Load aligned returns (already computed earlier)
returns = pd.read_feather(
    Data_processed / "mag7_returns.feather"
).set_index("date").sort_index()

# Align alpha with next-day returns
returns_forward = returns.shift(-1)
common_dates = alpha.index.intersection(returns_forward.index)

alpha_aligned = alpha.loc[common_dates]
returns_forward = returns_forward.loc[common_dates]

# Compute daily IC (Spearman rank correlation)
def daily_ic(alpha_row, return_row): 
    return spearmanr(alpha_row, return_row).correlation

ic_series = pd.Series(
    [
        daily_ic(alpha_aligned.loc[d], returns_forward.loc[d])
        for d in alpha_aligned.index
    ],
    index=alpha_aligned.index
)

# IC diagnostics
print("\nIC Summary")
print("------------")
print(f"Mean IC: {ic_series.mean():.4f}")
print(f"IC Std: {ic_series.std():.4f}")
print(f"IC t-stat: {ic_series.mean() / ic_series.std() * np.sqrt(len(ic_series)):.2f}")

# Load portfolio weights
weights = pd.read_feather(
    Data_processed / "mag7_portfolio_weights.feather"
).set_index("date").sort_index()

# Compute daily turnover 
turnover = weights.diff().abs().sum(axis=1)

# Turnover diagnostics
print("\nTurnover Summary")
print("------------------")
print(f"Average daily turnover: {turnover.mean():.2f}")
print(f"Median daily turnover: {turnover.median():.2f}")

# Save diagnostics
diagnostics = pd.DataFrame({
    "IC": ic_series,
    "turnover": turnover
})

diagnostics_file = Output_metrics / "signal_diagnostics.feather"
diagnostics.reset_index().to_feather(diagnostics_file)

print("\nDiagnostics saved to: ")
print(diagnostics_file.resolve())


# 10. Performance visualisations

# Load portfolio PnL
pnl_df = (pd.read_feather(
    Output_metrics / "portfolio_pnl.feather")
    .set_index("date")
    .sort_index()
)

# Load diagnostics
diagnostics_df = (pd.read_feather(
    Output_metrics / "signal_diagnostics.feather")
    .set_index("date")
    .sort_index()
)

# Equity Curve (Cumulative Returns)
fig_equity = px.line(
    pnl_df, 
    y="cumulative_returns",
    title="Portfolio Equity Curve (Log Returns)",
    labels={"value": "Cumulative Log Return", "date": "Date"} 
)

fig_equity.update_layout(
    template="plotly_dark",
    hovermode="x unified"
)

fig_equity.show()

# Daily Portfolio Returns (Distribution)
fig_returns_dist = px.histogram(
    pnl_df,
    x="portfolio_returns",
    nbins=50,
    title="Distribution of Daily Portfolio Returns"
)

fig_returns_dist.update_layout(
    template="plotly_dark", 
    bargap=0.05
)

fig_returns_dist.show()

# Information Coefficient - Time Series
fig_ic_ts = px.line(
    diagnostics_df, 
    y="IC", 
    title="Daily Information Coefficient (IC)",
    labels={"value": "IC", "date": "Date"}
)

fig_ic_ts.update_layout(
    template="plotly_dark",
    hovermode="x unified"
)

fig_ic_ts.show()

# Information Coefficient - Distribution
fig_ic_dist = px.histogram(
    diagnostics_df, 
    x="IC", 
    nbins=40,
    title="Distribution of Daily Information Coefficient"
)

fig_ic_dist.update_layout(
    template="plotly_dark",
    bargap=0.05
)

fig_ic_dist.show()

# Turnover Time Series
fig_turnover = px.line(
    diagnostics_df,
    y="turnover",
    title="Daily Portfolio Turnover",
    labels={"value": "Turnover", "date": "Date"}
)

fig_turnover.update_layout(
    template="plotly_dark", 
    hovermode="x unified"
)

fig_turnover.show()

# Rolling Sharpe
rolling_sharpe = (
    pnl_df["portfolio_returns"]
    .rolling(126)
    .mean()
    / pnl_df["portfolio_returns"].rolling(126).std()
) * np.sqrt(252)

fig_rolling_sharpe = px.line(
    rolling_sharpe,
    title="Rolling 6-Month Sharpe Ratio", 
    labels={"value": "Sharpe", "date": "Date"}
)

fig_rolling_sharpe.update_layout(
    template="plotly_dark",
    hovermode="x unified"
)

fig_rolling_sharpe.show()

# Consolidating all graphs in one dash
from plotly.offline import plot

dashboard_file = Output_figures / "mag7_momentum_dashboard.html"

with open(dashboard_file, "w", encoding="utf-8") as f:
    f.write("<html><head>")
    f.write("<meta charset='utf-8'>")
    f.write("<title>QIS Step-3: Mag7 Momentum Dashboard</title>")
    f.write("</head><body>")

    f.write("<h1>QIS Step-3 — Magnificent 7 Momentum Strategy</h1>")
    f.write("<p>Cross-sectional momentum (12–1), dollar-neutral, daily rebalanced.</p>")

    f.write("<h2>Equity Curve</h2>")
    f.write(plot(fig_equity, include_plotlyjs=True, output_type="div"))

    f.write("<h2>Daily Portfolio Return Distribution</h2>")
    f.write(plot(fig_returns_dist, include_plotlyjs=False, output_type="div"))

    f.write("<h2>Information Coefficient (Time Series)</h2>")
    f.write(plot(fig_ic_ts, include_plotlyjs=False, output_type="div"))

    f.write("<h2>Information Coefficient (Distribution)</h2>")
    f.write(plot(fig_ic_dist, include_plotlyjs=False, output_type="div"))

    f.write("<h2>Portfolio Turnover</h2>")
    f.write(plot(fig_turnover, include_plotlyjs=False, output_type="div"))

    f.write("<h2>Rolling 6-Month Sharpe Ratio</h2>")
    f.write(plot(fig_rolling_sharpe, include_plotlyjs=False, output_type="div"))

    f.write("</body></html>")

print("\nDashboard exported to:")
print(dashboard_file.resolve())
