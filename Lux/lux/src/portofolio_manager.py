import pickle
from lux import fetch_info
from lux import Interface
import datetime as dt
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.structural import UnobservedComponents
import scipy
import pandas as pd

def regress_input(y, N = 30):
    X = np.arange(0, N)[:, None]
    y = y.values[-N:]
    reg = Ridge().fit(X, y)
    score = reg.score(X, y)
    return reg.coef_[0], reg.intercept_, score


class Portofolio:
    def __init__(self):
        self.intfc = Interface()
        if os.path.exists('lux/database/portofolio.pkl'):
            self.load_data()
        else:
            self.prtf = {'tickers': []}

    def add_ticker(self, ticker):
        if ticker not in self.prtf['tickers']:
            print(f"Adding {ticker} to portofolio")
            self.prtf['tickers'].append(ticker)
            self.write_data()

    def remove_ticker(self, ticker):
        if ticker in self.prtf:
            self.prtf['tickers'].remove(ticker)
            self.write_data()
    
    def load_data(self):
        self.prtf = pickle.load(open('lux/database/portofolio.pkl', 'rb'))
    
    def write_data(self):
        pickle.dump(self.prtf, open('lux/database/portofolio.pkl', 'wb'))
    

    def get_stock_stats(self, N = 30):
        end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
        tmp_fetch = []
        self.intfc.make_table_moments()
        for ticker in self.prtf['tickers']:
            fetched = fetch_info(ticker, (end[0]-1,1,1), end)
            tmp_fetch.append(fetched[["Volume", "Adj Close"]])
            price_change_lowpass = fetched['Adj Close'].pct_change().rolling(10, win_type='gaussian').mean(std = 3)
            price_change_std = fetched['Adj Close'].pct_change().rolling(10, win_type='gaussian').std(std = 3)
    
            baselinevar = np.std(price_change_std)
            price_change_std = (price_change_std - np.mean(price_change_std))/np.std(price_change_std)
            minimum_historic_std = np.min(price_change_std)
            maximum_historic_std = np.max(price_change_std)
            mean_coeff, mean_intercept, scoremean = regress_input(price_change_lowpass, N = N)
            var_coeff, var_intercept, scorevar = regress_input(price_change_std, N = N)
            mean_now = price_change_lowpass[-1]
            var_now = price_change_std[-1]

            moments = {'mean_coeff': {'value': mean_coeff, 'output_text': '', 'color': 'green/red'}, 
                        'mean_intercept': {'value': mean_intercept, 'output_text': '', 'color': 'green/red'}, 
                        'mean_now': {'value': mean_now, 'output_text': '', 'color':  'green/red'}, 
                        'var_coeff': {'value': var_coeff, 'output_text': '', 'color': 'green/red'}, 
                        'var_intercept': {'value': var_intercept, 'output_text': '', 'color': 'yellow'}, 
                        'var_now': {'value': var_now, 'output_text': '', 'color': 'yellow'}, 
                        'minimum_historic_std': {'value': minimum_historic_std, 'output_text': '', 'color': 'yellow'}, 
                        'maximum_historic_std': {'value': maximum_historic_std, 'output_text': '', 'color': 'yellow'},
                        'baseline_historic_std': {'value': baselinevar, 'output_text': '', 'color': 'yellow'},
                        'scoremean': {'value': scoremean, 'output_text': '', 'color': 'yellow'},
                        'scorevar': {'value': scorevar, 'output_text': '', 'color': 'yellow'}
                        }



            ser = fetched['Adj Close'].dropna()


            # model = UnobservedComponents(ser.values,
            #                         level='fixed intercept',
            #                         seasonal=10,
            #                         freq_seasonal=[{'period': 100,
            #                                         'harmonics': 2}])
            # res_f = model.fit(disp=False)
            # print(res_f.summary())
            # res_f.plot_components()
            # plt.show()
            error = 99999
            for per in range(2, 120):
                result = seasonal_decompose(ser, model='additive', period = per)
                if np.sum(result.resid) < error:
                    error = np.sum(result.resid)
                    per_best = per
            print(per_best, error)
            result = seasonal_decompose(ser, model='additive', period = per_best)
            # print(result.trend)
            # print(result.seasonal)
            # print(result.resid)
            # print(result.observed)
            result.plot()
            plt.show()
            

            ## USED FOR TESTING ###
            # fig, ax = plt.subplots(3, 1)
            # ax[0].plot(fetched['Adj Close'], label = 'Price')
            # ax[0].plot(price_change_lowpass.index[-N], fetched['Adj Close'].values[-N], 'o', fillstyle='none')
            # ax[1].plot(price_change_lowpass, label = 'Lowpass 10 days', alpha = 0.4)
            # ax[1].plot(price_change_lowpass.index[-N], price_change_lowpass.values[-N], 'o', fillstyle='none')
            # ax[1].plot(fetched['Adj Close'].pct_change(), label = 'Returns', alpha = 0.4)
            # ax[2].plot(price_change_std, label = 'Lowpass standard deviation', alpha = 0.4)
            # ax[2].plot(price_change_std.index[-N], price_change_std.values[-N], 'o', fillstyle='none')
            # plt.suptitle(ticker)
            # ax[0].legend()
            # ax[1].legend()
            # ax[2].legend()
            # plt.show()
            self.intfc.add_row_moments(moments, ticker)
        self.intfc.console_print()


