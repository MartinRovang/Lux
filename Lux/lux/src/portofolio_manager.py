import pickle
from lux import fetch_info
import datetime as dt
import numpy as np
from rich.console import Console
from rich.table import Table
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy

def regress_input(y, N = 30):
    X = np.arange(0, N)[:, None]
    reg = LinearRegression().fit(X, y.values[-N:])
    return reg.coef_[0], reg.intercept_



class Portofolio:
    def __init__(self):
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
    
    def get_stock_stats(self):
        end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
        tmp_fetch = []
        console = Console()
        table = Table(title="Portofolio")
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("μθ30", style="cyan", no_wrap=True, justify="center")
        table.add_column("μb30", style="cyan", no_wrap=True, justify="center")
        table.add_column("μ30 now", style="cyan", no_wrap=True, justify="center")
        table.add_column("zσψ30", style="cyan", no_wrap=True, justify="center")
        table.add_column("zσb30", style="cyan", no_wrap=True, justify="center")
        table.add_column("zσ3 now", style="cyan", no_wrap=True, justify="center")
        table.add_column("zσ3 min/max|baseline.", style="cyan", no_wrap=True, justify="center")

        # table.add_column("30 days Var coeff", style="cyan", no_wrap=True)
        for ticker in self.prtf['tickers']:
            fetched = fetch_info(ticker, (end[0]-1,1,1), end)
            tmp_fetch.append(fetched[["Volume", "Adj Close"]])
            price_change_lowpass = fetched['Adj Close'].pct_change().rolling(30, win_type='gaussian').mean(std = 3)
            price_change_std = fetched['Adj Close'].pct_change().rolling(30, win_type='gaussian').std(std = 3)

            baselinevar = np.std(price_change_std)
            price_change_std = (price_change_std - np.mean(price_change_std))/np.std(price_change_std)
            minimum_historic_std = np.min(price_change_std)
            maximum_historic_std = np.max(price_change_std)
            mean_coeff, mean_intercept = regress_input(price_change_lowpass, N = 30)
            var_coeff, var_intercept = regress_input(price_change_std, N = 30)
            mean_now = price_change_lowpass[-1]
            var_now = price_change_std[-1]

            moments = {'mean_coeff': {'value': mean_coeff, 'output_text': '', 'color': 'green/red'}, 
                        'mean_intercept': {'value': mean_intercept, 'output_text': '', 'color': 'green/red'}, 
                        'mean_now': {'value': mean_now, 'output_text': '', 'color':  'green/red'}, 
                        'var_coeff': {'value': var_coeff, 'output_text': '', 'color': 'yellow'}, 
                        'var_intercept': {'value': var_intercept, 'output_text': '', 'color': 'yellow'}, 
                        'var_now': {'value': var_now, 'output_text': '', 'color': 'yellow'}, 
                        'minimum_historic_std': {'value': minimum_historic_std, 'output_text': '', 'color': 'yellow'}, 
                        'maximum_historic_std': {'value': maximum_historic_std, 'output_text': '', 'color': 'yellow'},
                        'baseline_historic_std': {'value': baselinevar, 'output_text': '', 'color': 'yellow'}
                        }


            # fig, ax = plt.subplots(3, 1)
            # ax[0].plot(fetched['Adj Close'], label = 'Price')
            # ax[1].plot(price_change_lowpass, label = 'Lowpass 30 days', alpha = 0.4)
            # ax[1].plot(fetched['Adj Close'].pct_change(), label = 'Returns', alpha = 0.4)
            # ax[2].plot(price_change_std, label = 'Lowpass standard deviation', alpha = 0.4)
            # ax[0].legend()
            # ax[1].legend()
            # ax[2].legend()
            # plt.show()
            for moment in moments:
                
                if moments[moment]['color'] == 'green/red':
                    if moments[moment]['value'] > 0:
                        color_output = 'green'
                    else:
                        color_output = 'red'
                else:
                    color_output = moments[moment]['color']
                moments[moment]['output_text'] = f"[bold {color_output}] {moments[moment]['value']:.5f} [/bold {color_output}] "
            table.add_row(ticker, moments['mean_coeff']['output_text'], moments['mean_intercept']['output_text'], moments['mean_now']['output_text'], moments['var_coeff']['output_text'], moments['var_intercept']['output_text'], moments['var_now']['output_text'], f"[{moments['minimum_historic_std']['output_text']}/{moments['maximum_historic_std']['output_text']}|{moments['baseline_historic_std']['output_text']}]")

        console.print(table)



        

    

    

    

    

        


