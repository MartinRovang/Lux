import pickle
from lux_dev import fetch_info
from lux_dev import Interface
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
import warnings
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

def regress_input(y, N = 30):
    """[summary]

    Args:
        y ([type]): [description]
        N (int, optional): [description]. Defaults to 30.

    Returns:
        [type]: [description]
    """
    X = np.arange(0, N)[:, None]
    y = y.values[-N:]
    reg = Ridge().fit(X, y)
    score = reg.score(X, y)
    return reg.coef_[0], reg.intercept_, score


class Portofolio:
    """[summary]

    Returns:
        [type]: [description]
    """    """"""
    def __init__(self):
        self.intfc = Interface()
        if os.path.exists('lux_dev/database/portofolio.pkl'):
            self.load_data()
        else:
            self.prtf = {'tickers': []}
        
        self.intfc.show_logo()

    def add_ticker(self, ticker, amount = 1, thresh=1):
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
        self.prtf = pickle.load(open('lux_dev/database/portofolio.pkl', 'rb'))
    
    def write_data(self):
        pickle.dump(self.prtf, open('lux_dev/database/portofolio.pkl', 'wb'))
    
    def normalize_max(self, df):
        return df/df.max()
    
    def normalize_z(self, df):
        return (df - df.mean())/df.std()
    

    def portofolio_metrics(self, portofolio, benchmark, weights = [], optimweights = True):
        """
        Args:
            portofolio ([type]): [description]
            benchmark ([type]): [description]
            weights (list, optional): [description]. Defaults to [].
            optimweights (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """        """"""
        # https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf
        # SIGMA = np.cov(portofolio)
        SIGMA = portofolio.cov()
        MU = portofolio.mean()
        # MU = np.mean(portofolio, axis = 1)*self.tradedays
        if optimweights == True:
            weights = np.ones(len(MU))
            SIGMA_INV = np.linalg.inv(SIGMA)
            weights = SIGMA_INV@weights/(weights.T@SIGMA_INV@weights) # Eq. (1.12) minimum variance portfolio weights


        mean = weights.T@MU
        mean = mean*self.tradedays
        std = np.sqrt(weights.T@SIGMA@weights)*np.sqrt(self.tradedays)
        std = std

        market_mean = np.mean(benchmark)*self.tradedays
        market_var = np.var(benchmark)*self.tradedays
        beta = std**2/market_var
        alpha = mean/market_mean

        # cov_market_index = np.cov(portofolio, benchmark)/np.var(benchmark)
        # cov_market_index = cov_market_index[0, 1:]
        # cov_market_index = cov_market_index.reshape(len(cov_market_index), 1)
        # beta_final = weights.T@cov_market_index

        output = {'mean': round(mean,5), 'std': round(std,5), 'beta': round(beta,5), 'sharpe': round(mean/std,5), 'weights': weights}
        return output
    

    def get_returns_from_prices(self, df):
        returns = df['Adj Close'].pct_change().dropna()
        return returns



    def iter_portofolio(self, portofolio_size, all_stock, benchmark, N):
        """[summary]

        Args:
            portofolio_size ([type]): [description]
            all_stock ([type]): [description]
            benchmark ([type]): [description]
            N ([type]): [description]

        Returns:
            [type]: [description]
        """
        # https://quant.stackexchange.com/questions/53992/is-this-methodology-for-finding-the-minimum-variance-portfolio-with-no-short-sel
        
        best_sharpe = 0
        worst_sharpe = 1
        worst_result = []
        if portofolio_size == None:
            portofolio_size_iter = 20
            for portsize in range(4, portofolio_size_iter+1):
                for iteration in track(list(range(N)), description=f'Optimizing [p {portsize}]...'):
                    current_portofolio = all_stock.sample(n = portofolio_size, replace = False, axis = 1)
                    result = self.portofolio_metrics(current_portofolio, benchmark)
                    result['portofolio'] = current_portofolio
                    if result['sharpe'] > best_sharpe and ((result['weights']>0).sum() == len(result['weights'])):
                        best_sharpe = result['sharpe']
                        result_best = result
                        
                    if (result['sharpe'] < worst_sharpe) and ((result['weights']>0).sum() == len(result['weights'])):
                        worst_sharpe = result['sharpe']
                        worst_result = result
        else:
            for iteration in track(list(range(N)), description='Optimizing...'):
                current_portofolio = all_stock.sample(n = portofolio_size, replace = False, axis = 1)

                result = self.portofolio_metrics(current_portofolio, benchmark)
                
                result['portofolio'] = current_portofolio
                if result['sharpe'] > best_sharpe and ((result['weights']>0).sum() == len(result['weights'])):
                    best_sharpe = result['sharpe']
                    result_best = result
                    
                if (result['sharpe'] < worst_sharpe) and ((result['weights']>0).sum() == len(result['weights'])):
                    worst_sharpe = result['sharpe']
                    worst_result = result
        
        return result_best, worst_result
    

    def print_optimized_port_results(self, result_best, worst_result, all_stock, symbols_not_loaded):
        """
        Args:
            result_best ([type]): [description]
            worst_result ([type]): [description]
            all_stock ([type]): [description]
            symbols_not_loaded ([type]): [description]
        """


        self.intfc.print_regular(f"----BEST PORTOFOLIO STOCKS [N = {len(result_best['portofolio'].columns.values)}]----", color = 'green')
        self.intfc.print_regular(f"{result_best['portofolio'].columns.values}", color = 'cyan')
        self.intfc.print_regular("----BEST WEIGHTS----", color = 'cyan')
        self.intfc.print_regular(f"{[round(weight, 4) for weight in result_best['weights']]}", color = 'green')
        self.intfc.print_regular("----ADDITIONAL METRICS----", color = 'cyan')
        self.intfc.print_regular(f"Expected annual return: {result_best['mean']*100:.3f} %", color = 'yellow')
        self.intfc.print_regular(f"Annual volatility: {result_best['std']*100:.3f} %", color = 'yellow')
        self.intfc.print_regular(f"Beta[S&P]: {result_best['beta']:.3f}", color = 'yellow')
        self.intfc.print_regular(f"Sharpe: {result_best['sharpe']}", color = 'yellow')

        self.intfc.print_regular(f"----WORST PORTOFOLIO STOCKS [N = {len(['portofolio'].columns.values)}]----", color = 'red')
        self.intfc.print_regular(f"{worst_result['portofolio'].columns.values}", color = 'red')
        self.intfc.print_regular("----BEST WEIGHTS----", color = 'cyan')
        self.intfc.print_regular(f"{[round(weight, 4) for weight in worst_result['weights']]}", color = 'red')
        self.intfc.print_regular("----ADDITIONAL METRICS----", color = 'cyan')
        self.intfc.print_regular(f"Expected annual return: {worst_result['mean']*100:.3f} %", color = 'yellow')
        self.intfc.print_regular(f"Annual volatility: {worst_result['std']*100:.3f} %", color = 'yellow')
        self.intfc.print_regular(f"Beta[S&P]: {worst_result['beta']:.3f}", color = 'yellow')
        self.intfc.print_regular(f"Sharpe: {worst_result['sharpe']}", color = 'yellow')

        self.intfc.print_regular("----Additional stats----", color = 'cyan')
        self.intfc.print_regular(f"Total stocks looked at: {str(all_stock.shape[1])}", color = 'yellow')
        self.intfc.print_regular(f"Time span: {self.tradedays}", color = 'yellow')
        self.intfc.print_regular("----Stocks not loaded due to being too young or error with loading data----", color = 'cyan')
        self.intfc.print_regular(f"{str(symbols_not_loaded)}", color = 'ywllow')
    


    def grab_symbols_from_yahoo(self, all_stocks_OSLO_symbols, end, benchmark):
        """

        Args:
            all_stocks_OSLO_symbols ([type]): [description]
            end ([type]): [description]
            benchmark ([type]): [description]

        Returns:
            [type]: [description]
        """        """"""
        benchmarklength = len(benchmark)
        i = 0
        symbols_according_to_index = []
        symbols_not_loaded = []
        for stocksymbol in track(all_stocks_OSLO_symbols, description='Grabbing latest data from all stocks...'):
            fetched = fetch_info(stocksymbol, (end[0]-self.years,end[1],end[2]), end)
            if type(fetched) == bool:
                symbols_not_loaded.append(stocksymbol)
                continue
            data = self.get_returns_from_prices(fetched[-self.tradedays*self.years:])
            data_prices = fetched['Adj Close'][-self.tradedays*self.years:]
            datalength = len(data)
            if datalength - benchmarklength == -1:
                benchmarklength = datalength
            if datalength == benchmarklength:
                symbols_according_to_index.append(stocksymbol)
                # data = data.values[None, :]
                if i < 1:
                    all_stock = data
                    all_data_prices = data_prices
                else:
                    all_stock = pd.concat([all_stock, data], axis = 1)
                    all_data_prices = pd.concat([all_data_prices, data_prices], axis = 1)
                i += 1
            else:
                # print(f'Failed to load data for TOO SHORT: {datalength} ticker: {stocksymbol}')
                symbols_not_loaded.append(stocksymbol)

        all_stock = all_stock.dropna()
        self.tradedays = len(all_stock)
        all_stock = pd.DataFrame(all_stock.values, index = all_stock.index, columns = symbols_according_to_index)
        all_data_prices = pd.DataFrame(all_data_prices.values, index = all_data_prices.index, columns = symbols_according_to_index)

        return all_stock, symbols_not_loaded, all_data_prices



    def make_optimized_portofolio(self, N = 100000, portofolio_size = 5, years = 3, tradedays = 250):
        #https://live.euronext.com/en/markets/oslo/equities/list
        """

        Args:
            N (int, optional): [description]. Defaults to 100000.
            portofolio_size (int, optional): [description]. Defaults to 5.
            years (int, optional): [description]. Defaults to 3.
            tradedays (int, optional): [description]. Defaults to 250.
        """        """"""
        self.years = years
        self.tradedays = tradedays
 
        self.intfc.print_regular("----Making optimized portofolio----", color = 'cyan')
        self.intfc.print_regular(f"Number of iterations: {N}", color = 'yellow')
        self.intfc.print_regular(f"Portofolio Size: {portofolio_size}", color = 'yellow')
        self.intfc.print_regular(f"Initiating...", color = 'cyan')

        all_stocks_OSLO = pd.read_csv('lux_dev/database/all_stocks_OSLO1.csv', encoding='latin', header=0, delimiter=';')
        all_stocks_OSLO['Symbol'] = all_stocks_OSLO['Symbol'] + '.OL'
        all_stocks_OSLO_symbols = all_stocks_OSLO['Symbol'].dropna()

        end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
        benchmark = fetch_info("^GSPC", (end[0]-years,end[1],end[2]), end)[-self.tradedays*self.years:]
        self.timespan = len(benchmark)
        benchmark = self.get_returns_from_prices(benchmark)

        all_stock, symbols_not_loaded = self.grab_symbols_from_yahoo(all_stocks_OSLO_symbols, end, benchmark)

        result_best, worst_result = self.iter_portofolio(portofolio_size, all_stock, benchmark, N)
        self.print_optimized_port_results(result_best, worst_result, all_stock, symbols_not_loaded)

    
    def simulate_portofolio(self, mean, variance, cash = 1000):
        """

        Args:
            mean ([type]): [description]
            variance ([type]): [description]
            cash (int, optional): [description]. Defaults to 1000.
        """        """"""
        #monte carlo simulation of future returns of portofolio
        #https://www.quantopian.com/posts/monte-carlo-simulation-of-portofolio-returns
        days = 60
        simulations = np.random.normal(mean/days, variance/np.sqrt(days), (10000, 60))
        print(simulations.shape)
        plt.plot(simulations)
        plt.show()


    def optimize_portofolio_already_given(self, weight_bounds=(0.05,1)):


        end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
        self.years = 1
        self.tradedays = 250
        benchmark = fetch_info("^GSPC", (end[0]-5,end[1],end[2]), end)[-self.tradedays*self.years:]
        benchmark = self.get_returns_from_prices(benchmark)
        self.tradedays = len(benchmark)
        all_stocks_OSLO_symbols = [stock for stock in self.prtf['tickers']]
        all_stock, symbols_not_loaded, all_data_prices = self.grab_symbols_from_yahoo(all_stocks_OSLO_symbols, end, benchmark)



        mu = expected_returns.mean_historical_return(all_data_prices, compounding= False)
        S = risk_models.sample_cov(all_data_prices)

        # Optimize for maximal Sharpe ratio
        ef = EfficientFrontier(mu, S, weight_bounds= weight_bounds)
        raw_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        # ef.save_weights_to_file("weights.csv")  # saves to file
        print(cleaned_weights)
        ef.portfolio_performance(verbose=True)


    
    def get_portofolio_stats(self, buy_date, buy_point, years: int = 1, weights: list = [1/3, 1/3, 1/3], init_cash = 10000) -> None:
        # https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf
        """

        Args:
            years (int, optional): [description]. Defaults to 1.
            weights (list, optional): [description]. Defaults to [1/3, 1/3, 1/3].

        Returns:
            [type]: [description]
        """
        buy_date = dt.datetime(*buy_date)
        weights = np.array(weights)
        end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
        self.years = years
        self.tradedays = 250
        benchmark = fetch_info("^GSPC", (end[0]-5,end[1],end[2]), end)[-self.tradedays*self.years:]
        benchmark = self.get_returns_from_prices(benchmark)
        self.tradedays = len(benchmark)
        all_stocks_OSLO_symbols = [stock for stock in self.prtf['tickers']]
        all_stock, symbols_not_loaded, all_data_prices = self.grab_symbols_from_yahoo(all_stocks_OSLO_symbols, end, benchmark)
        output = self.portofolio_metrics(all_stock, benchmark, weights = weights, optimweights = False)

        # self.intfc.print_regular(f"----PORTOFOLIO STATS [N = {all_stock.shape[1]}]----", color = 'cyan')
        # self.intfc.print_regular(f"Annual expected return: {round(output['mean']*100,2)}%", color = 'yellow')
        # self.intfc.print_regular(f"Annual devation: {round(output['std']*100,2)}%", color = 'yellow')
        # self.intfc.print_regular(f"Beta[S&P]: {output['beta']}", color = 'yellow')
        # self.intfc.print_regular(f"Sharpe ratio: {output['sharpe']}", color = 'yellow')


        window_period = 14
        num_sd = 2
        weighted_stocks = weights
        portofolio_value = (all_stock + 1).cumprod(axis = 0)
        portofolio_value = (portofolio_value*weighted_stocks).sum(axis = 1)
        mean_rolling = portofolio_value.rolling(window = window_period).mean().dropna()
        std_rolling = portofolio_value.rolling(window = window_period).std().dropna()


        print(output)
        print('Last point:', portofolio_value.values[-1])


        model = pm.auto_arima(mean_rolling, error_action='ignore', trace=True,
                  suppress_warnings=False, seasonal=False, n_fits = 1000 , max_p=100, max_d=2, max_q=100)
        forecasts, conf = model.predict(30, return_conf_int=True)  # predict N steps into the future


        date_future = pd.date_range(start=portofolio_value.index[-1], periods=30, freq='D')
        plt.plot(date_future, forecasts, '--', c='black', linewidth = 1, label = f'Projected {str(model)[:13]}', alpha = 0.4)
        plt.fill_between(date_future, conf[:, 1], conf[:, 0], color='black', alpha = 0.3)

        plt.plot(portofolio_value.index, portofolio_value.values, '-', color = 'blue', label = 'PF value', linewidth = 1 ,alpha = 1)
        plt.plot(mean_rolling.index, mean_rolling.values, '-', color = 'black', label = f'Mean {window_period} days', linewidth = 1 ,alpha = 0.7)
        plt.plot(mean_rolling.index, mean_rolling.values +num_sd*std_rolling.values, '-', color = 'black', linewidth = 1)
        plt.plot(mean_rolling.index, mean_rolling.values - num_sd*std_rolling.values, '-', color = 'black', linewidth = 1)
        plt.plot(buy_date, buy_point, 'o', color = 'green', label = 'Buy date', fillstyle = 'none')
        plt.plot(portofolio_value.index, np.repeat(buy_point, len(portofolio_value.index)), '-', color = 'green', label = 'profit line', alpha = 0.3)
        plt.plot(date_future, np.repeat(buy_point, len(date_future)), '-', color = 'green', alpha = 0.3)

        plt.fill_between(mean_rolling.index, mean_rolling.values - num_sd*std_rolling.values, mean_rolling.values + num_sd*std_rolling.values, color='black', alpha=0.4, label = f'{2}$\sigma$')
        plt.plot(portofolio_value.index[-1], portofolio_value.values[-1], '^', color = 'purple', label = 'Latest trading day', fillstyle = 'none')
        plt.title(f'Portofolio historical value [SR {round(output["sharpe"],2)}]')
        plt.ylabel('Portofolio value')
        plt.xlabel('Date')
        plt.legend(loc = 'upper left')
        plt.tight_layout()
        plt.savefig(f'lux_dev/database/portofolio_snapshots/jinx/{dt.datetime.now().strftime("%Y-%m-%d")}.pdf')
        plt.close()

        exit()



        sharpehistory = []
        window_period = 60
        windowed_stocks = all_stock.rolling(window = window_period)
        self.tradedays = window_period
        for i, window in enumerate(windowed_stocks):
            if i < window_period:
                continue
            window = window.dropna()
            self.tradedays = window.shape[0]
            output = self.portofolio_metrics(window, benchmark, weights = weights, optimweights = False)
            sharpehistory.append(output['sharpe'])

        sharpehistory_mean = pd.DataFrame(sharpehistory).rolling(window = window_period//3).mean().values[:, 0]
        sharpehistory_std = pd.DataFrame(sharpehistory).rolling(window = window_period//3).std().values[:, 0]

        sharpehistory = np.array(sharpehistory)
        tradedaycounter = list(range(0, len(sharpehistory)))

        # sharpehistory_rolling30 = pd.DataFrame(sharpehistory).rolling(window = 30, win_type='gaussian').mean(std=3)
        # tradedaycounter_30 = tradedaycounter[-len(sharpehistory_rolling30):]
        # sharpehistory_rolling10 = pd.DataFrame(sharpehistory).rolling(window = 30, win_type='gaussian').mean(std=3)
        # tradedaycounter_10 = tradedaycounter[-len(sharpehistory_rolling10):]

        
        # https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/bollinger-bands

        plt.plot(tradedaycounter, sharpehistory, '-', color = 'blue', label = 'Sharpe ratio', linewidth = 1 ,alpha = 0.6)
        plt.plot(tradedaycounter, np.repeat(1, len(tradedaycounter)), '--', color = 'red', label = 'Bad line')
        plt.plot(sharpehistory_mean, '--', color = 'black', label = 'Mean 60 day SR')
        plt.plot(sharpehistory_mean + ((window_period//3)/10)*sharpehistory_std, '-', color = 'black', linewidth = 1)
        plt.plot(tradedaycounter, sharpehistory_mean - ((window_period//3)/10)*sharpehistory_std, '-', color = 'black', linewidth = 1)
        plt.fill_between(tradedaycounter, sharpehistory_mean - ((window_period//3)/10)*sharpehistory_std, sharpehistory_mean + ((window_period//3)/10)*sharpehistory_std, color='black', alpha=0.3, label = 'Standard deviation')
        # plt.plot(tradedaycounter[-1], sharpehistory[-1], 'o', color = 'green', label = 'Latest trading day', fillstyle = 'none')
        plt.title('Sharpe ratio over time')
        plt.ylabel('Sharpe ratio')
        plt.xlabel('Latest trading days')
        # fig, ax = plt.subplots(1, 1)
        # ax[0].plot(tradedaycounter, sharpehistory, '-.', color = 'black', label = 'Sharpe ratio')
        # # ax[0].plot(tradedaycounter_30, sharpehistory_rolling30, '-.', color = 'red', label = 'Sharpe ratio 30 GMW')
        # # ax[0].plot(tradedaycounter_10, sharpehistory_rolling10, '-.', color = 'green', label = 'Sharpe ratio 10 GMW')
        # ax[0].plot(tradedaycounter[-1], sharpehistory[-1], 'o', color = 'green', label = 'Latest trading day', fillstyle = 'none')
        # ax[0].set_title('Sharpe ratio over time')
        # ax[0].set_ylabel('Sharpe ratio')
        # ax[0].set_xlabel('Latest trading days')
        
        # ax[1].plot(tradedaycounter, meanhistory, '-.', color = 'black', label = 'Expected return')
        # ax[1].plot(tradedaycounter[-1], meanhistory[-1], 'o', color = 'green', label = 'Latest trading day', fillstyle = 'none')
        # ax[1].set_title('Expected return over time')
        # ax[1].set_ylabel('Expected return')
        # ax[1].set_xlabel('Latest trading days')

        # ax[2].plot(tradedaycounter, stdhistory, '-.', color = 'black', label = 'Standard deviation')
        # ax[2].plot(tradedaycounter[-1], stdhistory[-1], 'o', color = 'green', label = 'Latest trading day', fillstyle = 'none')
        # ax[2].set_title('Standard deviation over time')
        # ax[2].set_ylabel('Standard deviation')
        # ax[2].set_xlabel('Latest trading days')


        # tradedaycounter_arima = list(range(30, tradedays_full -50+30))


        # model = pm.auto_arima(sharpehistory[:-50], error_action='ignore', trace=True,
        #           suppress_warnings=True, seasonal=True, n_fits = 1000)
        # forecasts, conf = model.predict(30, return_conf_int=True)  # predict N steps into the future

        # ax[0].plot(tradedaycounter_arima[-30:], forecasts, c='green')
        # ax[0].fill_between(tradedaycounter_arima[-30:], conf[:, 0], conf[:, 1], color='black', alpha=0.3, )

        plt.legend()
        plt.tight_layout()
        plt.show()




    def get_stock_stat(self, tickers = ['AFK.OL'], N = 30):
        """

        Args:
            tickers (list, optional): [description]. Defaults to ['AFK.OL'].
            N (int, optional): [description]. Defaults to 30.
        """        """"""


        end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
        col = ['Ticker', 'Trend[10/30]', 'Trendbias[10/30]', 'Confidence[10/30]']
        self.intfc.make_table(col, tl = 'Portofolio')
        for ticker in tickers:
            fetched = fetch_info(ticker, (end[0]-self.years,end[1],end[2]), end)
            ser = fetched['Adj Close'].dropna()


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
