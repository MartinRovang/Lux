from lux_dev import portofolio_manager
import numpy as np
from lux_dev import stocksaver
# stocksaver.grab_symbols_from_yahoo()
# stocksaver.main()



p = portofolio_manager.Portofolio()
# p.look_for_betterment(weights = [0.33494, 0.06668, 0.05, 0.05, 0.14806, 0.20032, 0.05, 0.05, 0.05])
# Samira
# p.add_ticker('MORG.OL')
# p.add_ticker('SDRL.OL')
# p.add_ticker('JAREN.OL')
# p.add_ticker('SOON.OL')
# p.add_ticker('POL.OL')

# Vi
# p.add_ticker('ALNG.OL')
# p.add_ticker('SOON.OL')
# p.add_ticker('JAREN.OL')
# p.add_ticker('KOG.OL')
# p.add_ticker('MORG.OL')

# p.get_stock_stat()
# Samira
# p.get_portofolio_stats(weights =  [0.29455796 ,0.00315625, 0.23164206, 0.37827824, 0.09236549])
# p.get_portofolio_stats(weights = [0.28197, 0.05, 0.08898, 0.43498, 0.14406])
# Vi
# p.get_portofolio_stats(years = 1, weights = [0.0174019, 0.34753948, 0.23776549, 0.07112895, 0.32616419])


# 10 stocks port test
p.add_ticker('SOON.OL')
p.add_ticker('JAREN.OL')
p.add_ticker('SNI.OL')
p.add_ticker('MOWI.OL')
p.add_ticker('POL.OL')
p.add_ticker('MORG.OL')
p.add_ticker('ODFB.OL')
p.add_ticker('NSKOG.OL')
p.add_ticker('BELCO.OL')
p.get_portofolio_stats(buy_date = (2022, 1 , 3), buy_point = 1.559 , years = 1, weights = [0.33494, 0.06668, 0.05, 0.05, 0.14806, 0.20032, 0.05, 0.05, 0.05])

# OrderedDict([('SOON.OL', 0.33494), ('JAREN.OL', 0.06668), ('SNI.OL', 0.05), ('MOWI.OL', 0.05), ('POL.OL', 0.14806), ('MORG.OL', 0.20032), ('ODFB.OL', 0.05), ('NSKOG.OL', 0.05), ('BELCO.OL', 0.05)])
# Expected annual return: 44.7%
# Annual volatility: 9.7%
# Sharpe Ratio: 4.39

# p.get_portofolio_stats(years = 1, weights = [0.3108, 0.1651, 0.0362, 0.0742, 0.0854, 0.0734, 0.2055, 0.0412, 0.0068, 0.0015])
# p.optimize_portofolio_already_given(weight_bounds=(0.05,1))
# p.get_portofolio_stats(weights = np.ones(len(p.prtf['tickers']))/len(p.prtf['tickers']))


# p.make_optimized_portofolio(N = 100000, portofolio_size = 10, years = 1)


# p.add_ticker('MPC')
# p.add_ticker('EQNR.OL')
# p.add_ticker('HSY')
# p.add_ticker('NEM')
# p.add_ticker('ROIV')
# p.add_ticker('LUMN')
# p.add_ticker('ATAX')
# p.add_ticker('BGCP')
# p.add_ticker('KTN')
# p.add_ticker('TECH.OL')
# p.get_portofolio_stats(buy_date = (2022, 1 , 3), buy_point = 1.559 , years = 1, weights = [0.039, 0.063, 0.377, 0.027, 0.031, 0.021, 0.067, 0.011, 0.327, 0.038])
##
# import pandas as pd
# from pypfopt import EfficientFrontier
# from pypfopt import risk_models
# from pypfopt import expected_returns
# import numpy as np 
# import datetime as dt
# import pandas_datareader as web

# # Read in price data
# # df = pd.read_csv("lux_dev/database/stock_prices.csv", parse_dates=True, index_col="date")

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
# print(S)

# # Optimize for maximal Sharpe ratio
# ef = EfficientFrontier(mu, S, weight_bounds=(0.05,1))
# raw_weights = ef.max_sharpe()
# print(raw_weights)
# cleaned_weights = ef.clean_weights()
# # ef.save_weights_to_file("weights.csv")  # saves to file
# print(cleaned_weights)
# ef.portfolio_performance(verbose=True)


#%%

# def make(weight, price):
#     out = (weight*100000)/price
#     return out


# print(make(0.07453796, 1.488))
# %%
