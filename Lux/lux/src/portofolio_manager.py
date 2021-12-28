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
import pmdarima as pm
from pmdarima import arima
from pmdarima import utils

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
        
        self.intfc.show_logo()

    def add_ticker(self, ticker, amount = 1):
        if ticker not in self.prtf['tickers']:
            print(f"Adding {ticker} to portofolio")
            self.prtf['tickers'].append(ticker)
            self.write_data()

    def remove_ticker(self, ticker):
        if ticker in self.prtf:
            print(f"Removing {ticker} from portofolio")
            self.prtf['tickers'].remove(ticker)
            self.write_data()
    
    def load_data(self):
        self.prtf = pickle.load(open('lux/database/portofolio.pkl', 'rb'))
    
    def write_data(self):
        pickle.dump(self.prtf, open('lux/database/portofolio.pkl', 'wb'))
    
    def normalize_max(self, df):
        return df/df.max()
    
    def normalize_z(self, df):
        return (df - df.mean())/df.std()
    
    def get_portofolio_stats(self):
        # https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf
        end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
        MU = []
        weights = np.ones(len(self.prtf['tickers']))
        fetched_vix = fetch_info("^VIX", (end[0]-1,1,1), end)
        for i, ticker in enumerate(self.prtf['tickers']):
            fetched = fetch_info(ticker, (end[0]-1,1,1), end)
            ser = fetched['Adj Close'].pct_change().dropna()
            MU.append(np.mean(ser))
            ser = ser.values[None, :]
            if i < 1:
                all_stock = ser
            else:
                all_stock = np.concatenate((all_stock, ser), axis = 0)

        SIGMA = np.cov(all_stock)
        MU = np.array(MU)

        vix_std = np.std(fetched_vix['Adj Close'].pct_change().dropna())
        mean = weights@MU
        std = np.sqrt(weights.T@SIGMA@weights)
        print(f"Portofolio stats:")
        print("Expected return day to day: ", round(mean*100,2),"%")
        print("Expected devation day to day:", round(std*100,2),"%")
        print("Beta[VIX]: ", round(std/vix_std, 2))
        print("Sharpe ratio: ", round(mean/std, 2))



    def get_stock_stat(self, tickers = ['AFK.OL'] , N = 30):
        end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
        col = ['Ticker', 'Trend[10/30]', 'Trendbias[10/30]', 'Confidence[10/30]']
        self.intfc.make_table(col, tl = 'Portofolio')
        for ticker in tickers:
            fetched = fetch_info(ticker, (end[0]-1,1,1), end)
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


            # model = pm.auto_arima(ser, error_action='ignore', trace=True,
            #           suppress_warnings=True, seasonal=True, m=13)
            # forecasts, conf = model.predict(30, return_conf_int=True)  # predict N steps into the future
            # trainlen = len(ser)
            # predlen = 30
            # x_axis = np.arange(trainlen + predlen)
            # plt.plot(x_axis[:len(ser)], ser, c='blue')
            # plt.plot(x_axis[-30:], forecasts, c='green')
            # plt.fill_between(x_axis[-30:], conf[:, 0], conf[:, 1], color='orange', alpha=0.3)
            # plt.show()

            # figure_kwargs = {'figsize': (6, 6)}
            # decomposed = arima.decompose(ser.values, 'additive', m=60)
            # axes = utils.decomposed_plot(decomposed, figure_kwargs=figure_kwargs,
            #                  show=False)
            # axes[0].set_title(ticker)
            # plt.show()

            #ser = self.normalize_max(ser).dropna()
            error = 99999999
            for per in range(10, 60):
                result = seasonal_decompose(ser, model='additive', period = per, two_sided = False)
                get_sample = result.trend[-N:]
                if np.std(get_sample) < error:
                    error = np.std(get_sample)
                    per_best = per
            result = seasonal_decompose(ser, model='additive', period = per_best, two_sided = False)
            # result.plot()
            # plt.suptitle(ticker)
            # plt.show()

            pct_trend = result.trend.pct_change()
            trend_coeff10, trend_intercept10, trend_score10 = regress_input(pct_trend, N = 10)
            trend_coeff30, trend_intercept30, trend_score30 = regress_input(pct_trend, N = 30)

            # plt.plot(pct_trend)
            # plt.title(ticker)
            # plt.show()
            # print(result.trend)
            # print(result.seasonal)
            # print(result.resid)
            # print(result.observed)

            trend_coeff10_text = self.intfc.redgreencolor(trend_coeff10)
            trend_intercept10_text = self.intfc.redgreencolor(trend_intercept10)
            trend_score10_text = self.intfc.colorify(trend_score10, color = 'yellow', type = 'float')
            trend_coeff30_text = self.intfc.redgreencolor(trend_coeff30)
            trend_intercept30_text = self.intfc.redgreencolor(trend_intercept30)
            trend_score30 = self.intfc.colorify(trend_score30, color = 'yellow', type = 'float')

            trend_coeff = f"[{trend_coeff10_text}/{trend_coeff30_text}]"
            trend_inter = f"[{trend_intercept10_text}/{trend_intercept30_text}]"
            trend_score = f"[{trend_score10_text}/{trend_score30}]"

            rows = [ticker,
                    trend_coeff,
                    trend_inter,
                    trend_score,
                    ]
            self.intfc.add_row(rows)
        self.intfc.console_print()


