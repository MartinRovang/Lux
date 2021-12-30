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
from rich.progress import track
import scipy
import pandas as pd
import time


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
    

    def portofolio_metrics(self, portofolio, benchmark, weights = False):
        # https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf
        SIGMA = np.cov(portofolio)
        MU = np.mean(portofolio, axis = 1)*self.timespan
        MU = MU.reshape(len(MU), 1)

        if weights == False:
            weights = np.ones(len(portofolio))
            SIGMA_INV = np.linalg.inv(SIGMA)
            weights = SIGMA_INV@weights/(weights.T@SIGMA_INV@weights) # Eq. (1.12) minimum variance portfolio weights
            weights = weights.reshape(len(weights), 1)


        benchmark_std = np.std(benchmark)
        mean = weights.T@MU
        mean = mean[0][0]

        std = np.sqrt(weights.T@SIGMA@weights)*np.sqrt(self.timespan)
        std = std[0][0]

        beta_list = []
        for stock in portofolio:
            cov = np.cov(stock, benchmark)[0][1]
            beta = cov/benchmark_std**2
            beta_list.append(beta)
        beta_list = np.array(beta_list).reshape(len(beta_list), 1)
        beta_final = weights.T@beta_list
        beta_final = beta_final[0][0]



        output = {'mean': round(mean,5), 'std': round(std,5), 'beta': round(beta_final,5), 'sharpe': round(mean/std,5), 'weights': weights}
        return output
    

    def get_returns_from_prices(self, df):
        if self.timespan != None:
            return df['Adj Close'].pct_change().dropna(how="all")# + 1).cumprod()[-timespan:] - 1
        else:
            return df['Adj Close'].pct_change().dropna(how="all")# + 1).cumprod()-1#.cumsum()-1


    def iter_portofolio(self, symbols_according_to_index, portofolio_size, all_stock, benchmark, N):
        # https://quant.stackexchange.com/questions/53992/is-this-methodology-for-finding-the-minimum-variance-portfolio-with-no-short-sel
        
        symbols_according_to_index = np.array(symbols_according_to_index)
        random_index_sample = np.arange(0, len(symbols_according_to_index))
        best_sharpe = 0
        worst_sharpe = 0
        worst_result = []
        portofolio_list_worst = []
        if portofolio_size == None:
            portofolio_size_iter = 20
            for portsize in range(4, portofolio_size_iter+1):
                for iteration in track(list(range(N)), description=f'Optimizing [p {portsize}]...'):
                    current_random_sample = np.random.choice(random_index_sample.copy(), size = portsize, replace=False)
                    current_random_sample = np.array(current_random_sample)
                    current_portofolio = all_stock.copy()[current_random_sample]
                    current_portofolio_symbols = symbols_according_to_index.copy()[current_random_sample]
                    result = self.portofolio_metrics(current_portofolio, benchmark)
                    if result['sharpe'] > best_sharpe:
                        best_sharpe = result['sharpe']
                        result_best = result
                        portofolio_list_best = current_portofolio_symbols
                    
                    if result['sharpe'] < worst_sharpe:
                        worst_sharpe = result['sharpe']
                        worst_result = result
                        portofolio_list_worst = current_portofolio_symbols
        else:
            for iteration in track(list(range(N)), description='Optimizing...'):
                current_random_sample = np.random.choice(random_index_sample.copy(), size = portofolio_size, replace=False)
                current_random_sample = np.array(current_random_sample)
                current_portofolio = all_stock.copy()[current_random_sample]
                current_portofolio_symbols = symbols_according_to_index.copy()[current_random_sample]

                result = self.portofolio_metrics(current_portofolio, benchmark)
                if result['sharpe'] > best_sharpe and ((result['weights']>0).sum() == len(result['weights'])):
                    best_sharpe = result['sharpe']
                    result_best = result
                    portofolio_list_best = current_portofolio_symbols
                
                if (result['sharpe'] < worst_sharpe) and ((result['weights']>0).sum() == len(result['weights'])):
                    worst_sharpe = result['sharpe']
                    worst_result = result
                    portofolio_list_worst = current_portofolio_symbols
        
        return result_best, portofolio_list_best, worst_result, portofolio_list_worst
    

    def print_optimized_port_results(self, portofolio_list_best, result_best, portofolio_list_worst, worst_result, all_stock, symbols_not_loaded):

        self.intfc.print_regular(f"----BEST PORTOFOLIO STOCKS [N = {len((portofolio_list_best))}]----", color = 'green')
        self.intfc.print_regular(f"{str(portofolio_list_best)}", color = 'cyan')
        self.intfc.print_regular("----BEST WEIGHTS----", color = 'cyan')
        self.intfc.print_regular(f"{result_best['weights']}", color = 'green')
        self.intfc.print_regular("----ADDITIONAL METRICS----", color = 'cyan')
        self.intfc.print_regular(f"Expected annual return: {result_best['mean']*100:.3f} %", color = 'yellow')
        self.intfc.print_regular(f"Annual volatility: {result_best['std']*100:.3f} %", color = 'yellow')
        self.intfc.print_regular(f"Beta[S&P]: {result_best['beta']:.3f}", color = 'yellow')
        self.intfc.print_regular(f"Sharpe: {result_best['sharpe']}", color = 'yellow')

        self.intfc.print_regular(f"----WORST PORTOFOLIO STOCKS [N = {len((portofolio_list_worst))}]----", color = 'red')
        self.intfc.print_regular(f"{str(portofolio_list_worst)}", color = 'red')
        self.intfc.print_regular("----BEST WEIGHTS----", color = 'cyan')
        self.intfc.print_regular(f"{worst_result['weights']}", color = 'red')
        self.intfc.print_regular("----ADDITIONAL METRICS----", color = 'cyan')
        self.intfc.print_regular(f"Expected annual return: {worst_result['mean']*100:.3f} %", color = 'yellow')
        self.intfc.print_regular(f"Annual volatility: {worst_result['std']*100:.3f} %", color = 'yellow')
        self.intfc.print_regular(f"Beta[S&P]: {worst_result['beta']:.3f}", color = 'yellow')
        self.intfc.print_regular(f"Sharpe: {worst_result['sharpe']}", color = 'yellow')

        self.intfc.print_regular("----Additional stats----", color = 'cyan')
        self.intfc.print_regular(f"N stocks: {str(len(all_stock))}", color = 'yellow')
        self.intfc.print_regular(f"Time span: {self.timespan}", color = 'yellow')
        self.intfc.print_regular("----Stocks not loaded due to being too young or error with loading data----", color = 'cyan')
        self.intfc.print_regular(f"{str(symbols_not_loaded)}", color = 'ywllow')
    


    def grab_symbols_from_yahoo(self, all_stocks_OSLO_symbols, end, benchmark):
        benchmarklength = len(benchmark)
        i = 0
        symbols_according_to_index = []
        symbols_not_loaded = []
        for stocksymbol in track(all_stocks_OSLO_symbols, description='Grabbing latest data from all stocks...'):
            fetched = fetch_info(stocksymbol, (end[0]-1,end[1],end[2]), end)
            if type(fetched) == bool:
                    symbols_not_loaded.append(stocksymbol)
                    continue
            data = self.get_returns_from_prices(fetched)
            datalength = len(data)
            if datalength == benchmarklength:
                symbols_according_to_index.append(stocksymbol)
                data = data.values[None, :]
                if i < 1:
                    all_stock = data
                else:
                    all_stock = np.concatenate((all_stock, data), axis = 0)
                i += 1
            else:
                # print(f'Failed to load data for TOO SHORT: {datalength} ticker: {stocksymbol}')
                symbols_not_loaded.append(stocksymbol)
            # if i >= 10:
            #     break
        return all_stock, symbols_according_to_index, symbols_not_loaded

    

    def make_optimized_portofolio(self, N = 100000, portofolio_size = 5, timespan = 252):
        #https://live.euronext.com/en/markets/oslo/equities/list
        self.timespan = timespan
 
        self.intfc.print_regular("----Making optimized portofolio----", color = 'cyan')
        self.intfc.print_regular(f"Number of iterations: {N}", color = 'yellow')
        self.intfc.print_regular(f"Portofolio Size: {portofolio_size}", color = 'yellow')
        self.intfc.print_regular(f"Initiating...", color = 'cyan')

        all_stocks_OSLO = pd.read_csv('lux/database/all_stocks_OSLO1.csv', encoding='latin', header=0, delimiter=';')
        all_stocks_OSLO['Symbol'] = all_stocks_OSLO['Symbol'] + '.OL'
        all_stocks_OSLO_symbols = all_stocks_OSLO['Symbol'].dropna()

        end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
        benchmark = fetch_info("^GSPC", (end[0]-1,end[1],end[2]), end)
        benchmark = self.get_returns_from_prices(benchmark)

        all_stock, symbols_according_to_index, symbols_not_loaded = self.grab_symbols_from_yahoo(all_stocks_OSLO_symbols, end, benchmark)

        result_best, portofolio_list_best, worst_result, portofolio_list_worst = self.iter_portofolio(symbols_according_to_index, portofolio_size, all_stock, benchmark, N)
        self.print_optimized_port_results(portofolio_list_best, result_best, portofolio_list_worst, worst_result, all_stock, symbols_not_loaded)

    
    def simulate_portofolio(self, mean, variance, cash = 1000):
        #monte carlo simulation of future returns of portofolio
        #https://www.quantopian.com/posts/monte-carlo-simulation-of-portofolio-returns
        days = 60
        simulations = np.random.normal(mean/days, variance/np.sqrt(days), (10000, 60))
        print(simulations.shape)
        plt.plot(simulations)
        plt.show()

        pass



    
    def get_portofolio_stats(self):
        # https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf
        end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
        weights = np.ones(len(self.prtf['tickers']))/len(self.prtf['tickers'])
        # ^VIX
        benchmark = fetch_info("^GSPC", (end[0]-1,1,1), end)
        for i, ticker in enumerate(self.prtf['tickers']):
            fetched = fetch_info(ticker, (end[0]-1,1,1), end)
            self.get_returns_from_prices(ser)
            self.timespan = None
            # MU.append(np.mean(ser))
            ser = ser.values[None, :]
            if i < 1:
                all_stock = ser
            else:
                all_stock = np.concatenate((all_stock, ser), axis = 0)

        result = self.portofolio_metrics(all_stock, weights, benchmark)
        print(f"Portofolio stats:")
        print("Expected return day to day: ", result['mean']*100,"%")
        print("Expected devation day to day:", result['std']*100,"%")
        print("Beta[S&P]: ", result['beta'])
        print("Sharpe ratio: ", result['sharpe'])



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




# %%
