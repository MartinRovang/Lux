from lux import portofolio_manager

p = portofolio_manager.Portofolio()
# p.add_ticker('AFK.OL')
# p.add_ticker('SALM.OL')
# p.add_ticker('YAR.OL')
# p.add_ticker('EQNR.OL')
# p.add_ticker('TEL.OL')
# p.add_ticker('MPCC.OL')
# p.add_ticker('MOWI.OL')
# p.get_stock_stat()
# p.get_portofolio_stats()
p.make_optimized_portofolio(N = 100000, portofolio_size = 5, timespan = 230)



###
# import pandas as pd
# from pypfopt import EfficientFrontier
# from pypfopt import risk_models
# from pypfopt import expected_returns
# import numpy as np 
# import datetime as dt
# import pandas_datareader as web

# # Read in price data
# # df = pd.read_csv("lux/database/stock_prices.csv", parse_dates=True, index_col="date")

# end = dt.datetime.now()
# start = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
# start = (start[0]-1,start[1],start[2])
# start = dt.datetime(*start)
# # print(end)
# # print(start)
# all = []
# out = web.DataReader('SOON.OL', 'yahoo', start, end).rename(columns={'Adj Close': 'SOON.OL'})['SOON.OL'][-252:]
# out1 = web.DataReader('MORG.OL', 'yahoo', start, end).rename(columns={'Adj Close': 'MORG.OL'})['MORG.OL'][-252:]

# out2 = web.DataReader('RIVER.OL', 'yahoo', start, end).rename(columns={'Adj Close': 'RIVER.OL'})['RIVER.OL'][-252:]
# out3 = web.DataReader('EQNR.OL', 'yahoo', start, end).rename(columns={'Adj Close': 'EQNR.OL'})['EQNR.OL'][-252:]

# out4 = web.DataReader('QFUEL.OL', 'yahoo', start, end).rename(columns={'Adj Close': 'QFUEL.OL'})['QFUEL.OL'][-252:]

# outt = pd.concat([out, out1, out2, out3, out4], axis = 1)
# # ['SOON.OL' 'MORG.OL' 'RIVER.OL' 'EQNR.OL' 'QFUEL.OL'
# print(outt)



# # Calculate expected returns and sample covariance
# mu = expected_returns.mean_historical_return(outt, compounding= False)
# S = risk_models.sample_cov(outt)

# print(mu)
# # print(S)

# # Optimize for maximal Sharpe ratio
# ef = EfficientFrontier(mu, S)
# raw_weights = ef.max_sharpe()
# print(raw_weights)
# cleaned_weights = ef.clean_weights()
# # ef.save_weights_to_file("weights.csv")  # saves to file
# print(cleaned_weights)
# ef.portfolio_performance(verbose=True)


#%%

def make(weight, price):
    out = (weight*100000)/price
    return out


print(make(0.32616419, 444))
# %%
