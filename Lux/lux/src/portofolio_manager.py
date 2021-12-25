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
        table.add_column("30 days moments", style="cyan", no_wrap=True, justify="center",)
        # table.add_column("30 days Var coeff", style="cyan", no_wrap=True)
        for ticker in self.prtf['tickers']:
            fetched = fetch_info(ticker, (end[0]-1,1,1), end)
            tmp_fetch.append(fetched[["Volume", "Adj Close"]])
            price_change_lowpass = fetched['Adj Close'].pct_change().rolling(30, win_type='gaussian').mean(std = 3)
            price_change_std = fetched['Adj Close'].pct_change().rolling(30, win_type='gaussian').std(std = 3)


            price_change_std = (price_change_std - np.mean(price_change_std))/np.std(price_change_std)
            minimum_historic_std = np.min(price_change_std)
            maximum_historic_std = np.max(price_change_std)
            mean_coeff, mean_intercept = regress_input(price_change_lowpass, N = 30)
            var_coeff, var_intercept = regress_input(price_change_std, N = 30)
            mean_now = price_change_lowpass[-1]
            var_now = price_change_std[-1]
            moments = [mean_coeff, mean_intercept, mean_now, var_coeff, var_now, var_intercept]


            fig, ax = plt.subplots(3, 1)
            ax[0].plot(fetched['Adj Close'], label = 'Price')
            ax[1].plot(price_change_lowpass, label = 'Lowpass 30 days', alpha = 0.4)
            ax[1].plot(fetched['Adj Close'].pct_change(), label = 'Returns', alpha = 0.4)
            ax[2].plot(price_change_std, label = 'Lowpass standard deviation', alpha = 0.4)
            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
            plt.show()
            colors = {}
            for moment in moments:
                if moment > 0:
                    colors[moment] = 'green'
                elif moment < 0:
                    colors[moment] = 'red'
                else:
                    colors[moment] = 'yellow'
            table.add_row(f"{ticker}", f"[bold {colors[moments[0]]}] {moments[0]:.5f} [/bold {colors[moments[0]]}](μψ30) [bold {colors[moments[1]]}] {moments[1]:.5f}[/bold {colors[moments[1]]}] [[bold {colors[moments[2]]}] {moments[2]:.5f} [/bold {colors[moments[2]]}]](μθ30) [bold {colors[moments[3]]}]{moments[3]:.5f} [/bold {colors[moments[3]]}](zσψ30) [bold {colors[moments[5]]}] {moments[5]:.5f} [/bold {colors[moments[5]]}][{minimum_historic_std:.5f}/{maximum_historic_std:.5f}][[bold {colors[moments[4]]}] {moments[4]:.5f} [/bold {colors[moments[4]]}]](zσθ30)")

        console.print(table)



        

    

    

    

    

        


