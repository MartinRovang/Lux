#%%
import numpy as np
import pandas as pd
import yfinance as yf
import riskfolio as rp
import matplotlib.pyplot as plt




import pandas as pd
from pypfopt import EfficientFrontier, plotting
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions

# Date range
start = '2021-01-01'
end = '2022-03-24'

# Tickers of assets
tickers = ['POL.OL', 'SNI.OL', 'BELCO.OL', 'NSKOG.OL', 'ODFB.OL', 'MORG.OL', 'MOWI.OL', 'JAREN.OL',
           'SOON.OL', 'SACAM.OL', 'NRS.OL', 'VOLUE.OL', 'SATS.OL', 'NOD.OL', 'AGAS.OL', 'WEED.TO', 'KOG.OL']
tickers.sort()

# Downloading the data
data = yf.download(tickers, start = start, end = end)
data = data['Adj Close']
# data = data.loc[:,('Adj Close', slice(None))]
data.columns = tickers
# assets = data.pct_change().dropna()

# Read in price data
#df = pd.read_csv("tests/resources/stock_prices.csv", parse_dates=True, index_col="date")

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(data)
S = risk_models.sample_cov(data)

# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
# raw_weights = ef.max_sharpe()

ef.add_objective(objective_functions.L2_reg, gamma=1)
ef.max_sharpe()
ef.min_volatility()

cleaned_weights = ef.clean_weights()
# ef.save_weights_to_file("weights.csv")  # saves to file
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

#%%
import copy
fig, ax = plt.subplots()
ef_max_sharpe = copy.deepcopy(ef)
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# Find the tangency portfolio
ef_max_sharpe.max_sharpe()
ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

# Generate random portfolios
n_samples = 10000
w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
rets = w.dot(ef.expected_returns)
stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Output
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=200)
plt.show()
#%%










#%%

yf.pdr_override()

# Date range
start = '2021-01-01'
end = '2022-03-24'

# Tickers of assets
tickers = ['POL.OL', 'SNI.OL', 'BELCO.OL', 'NSKOG.OL', 'ODFB.OL', 'MORG.OL', 'MOWI.OL', 'JAREN.OL',
           'SOON.OL', 'SACAM.OL', 'NRS.OL', 'VOLUE.OL', 'SATS.OL', 'NOD.OL', 'AGAS.OL', 'WEED.TO', 'KOG.OL']
tickers.sort()

# Downloading the data
data = yf.download(tickers, start = start, end = end)
data = data.loc[:,('Adj Close', slice(None))]
data.columns = tickers
assets = data.pct_change().dropna()

Y = assets

# Creating the Portfolio Object
port = rp.Portfolio(returns=Y)

# To display dataframes values in percentage format
pd.options.display.float_format = '{:.4%}'.format

# Choose the risk measure
rm = 'MV'  # Standard Deviation

# Estimate inputs of the model (historical estimates)
method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

# Estimate the portfolio that maximizes the risk adjusted return ratio
w = port.optimization(model='Classic', rm=rm, obj='Sharpe', rf=0.0, l=0, hist=True)
# Estimate points in the efficient frontier mean - semi standard deviation
ws = port.efficient_frontier(model='Classic', rm=rm, points=20, rf=0, hist=True)

#%%
nav=port.nav
mu = port.mu
cov = port.cov

ax1 = rp.plot_table(returns=Y, w=w, MAR=0, alpha=0.05, ax=None)
plt.show()
ax2 = rp.plot_drawdown(nav=nav, w=w, alpha=0.05, height=8, width=10, ax=None)
plt.show()
ax3 = rp.plot_pie(w=w, title='Portafolio', height=6, width=10,
                 cmap="tab20", ax=None)
plt.show()

# ax4 = rp.plot_risk_con(w=w, cov=cov, returns=Y, rm=rm,
#                       rf=0, alpha=0.05, color="tab:blue", height=6,
#                       width=10, t_factor=252, ax=None)
# plt.show()
ax5 = rp.plot_hist(returns=Y, w=w, alpha=0.05, bins=50, height=6,
                  width=10, ax=None)
plt.show()
ax6 = rp.plot_range(returns=Y, w=w, alpha=0.05, a_sim=100, beta=None,
                b_sim=None, bins=50, height=6, width=10, ax=None)
plt.show()
ax7 = rp.plot_clusters(returns=Y,
                      linkage='ward', k=None, max_k=10,
                      leaf_order=True, dendrogram=True, ax=None)
plt.show()
label = 'Max Risk Adjusted Return Portfolio'
mu = port.mu
cov = port.cov
returns = port.returns

ax = rp.plot_frontier(w_frontier=ws, mu=mu, cov=cov, returns=returns,
                       rm=rm, rf=0, alpha=0.05, cmap='viridis', w=w,
                       label=label, marker='*', s=16, c='r',
                       height=6, width=10, t_factor=252, ax=None)
plt.show()
# %%
w.shape
# %%
