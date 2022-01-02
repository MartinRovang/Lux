import pandas as pd
import datetime as dt
from lux import fetch_info
from rich.progress import track
import numpy as np
import pickle
import matplotlib.pyplot as plt


def get_returns_from_prices(df):
    return df['Adj Close'].pct_change().dropna()# + 1).cumprod()[-timespan:] - 1

def grab_symbols_from_yahoo():
        """

        Args:
            all_stocks_OSLO_symbols ([type]): [description]
            end ([type]): [description]
            benchmark ([type]): [description]

        Returns:
            [type]: [description]
        """        """"""

        all_stocks_OSLO = pd.read_csv('lux/database/all_stocks_OSLO1.csv', encoding='latin', header=0, delimiter=';')
        all_stocks_OSLO['Symbol'] = all_stocks_OSLO['Symbol'] + '.OL'
        all_stocks_OSLO = pd.concat([all_stocks_OSLO , pd.DataFrame(["^GSPC"], columns=['Symbol'])])
        all_stocks_OSLO_symbols = all_stocks_OSLO['Symbol'].dropna()
        years = 3
        tradedays = 250

        end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
        i = 0
        symbols_according_to_index = []
        symbols_not_loaded = []
        for stocksymbol in track(all_stocks_OSLO_symbols, description='Grabbing latest data from all stocks...'):
            fetched = fetch_info(stocksymbol, (end[0]-years,end[1],end[2]), end)
            if type(fetched) == bool:
                symbols_not_loaded.append(stocksymbol)
                continue
            
            data = get_returns_from_prices(fetched[-tradedays*years:])
            symbols_according_to_index.append(stocksymbol)
            if i < 1:
                all_stock = data
            else:
                all_stock = pd.concat([all_stock, data], axis = 1)
            i += 1
        all_stock = pd.DataFrame(all_stock.values, index = all_stock.index, columns = symbols_according_to_index)
        pickle.dump(all_stock, open(f'lux/database/stockprices/all_stock_vals{end}.pkl', 'wb'))



def portofolio_metrics(portofolio, benchmark, optimweights = True):
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


    tradedays = 250

    mean = weights.T@MU
    mean = mean*tradedays
    std = np.sqrt(weights.T@SIGMA@weights)*np.sqrt(tradedays)
    std = std

    market_mean = np.mean(benchmark)*tradedays
    market_var = np.var(benchmark)*tradedays
    beta = std**2/market_var
    alpha = mean/market_mean

    # cov_market_index = np.cov(portofolio, benchmark)/np.var(benchmark)
    # cov_market_index = cov_market_index[0, 1:]
    # cov_market_index = cov_market_index.reshape(len(cov_market_index), 1)
    # beta_final = weights.T@cov_market_index

    output = {'mean': round(mean,5), 'std': round(std,5), 'beta': round(beta,5), 'sharpe': round(mean/std,5), 'weights': weights}
    return output


def main():
    # grab_symbols_from_yahoo()
    data = pickle.load(open('lux/database/stockprices/all_stock_vals(2022, 1, 1).pkl', 'rb'))
    data_benchmark = data["^GSPC"]
    data_all = data.drop("^GSPC", axis=1)
    data_benchmark = data_benchmark.dropna()[-250:]
    data_all = data_all[-250:]
    leng = data_all.shape[0]
    data_all = data_all.dropna(axis = 1, thresh= leng - 10)
    all_port = {}
    for iteration in track(range(0, 10000)):
        current_portofolio = data_all.sample(n = 5, replace = False, axis = 1)
        out = portofolio_metrics(current_portofolio, data_benchmark)
        out['portofolio'] = current_portofolio.columns
        all_port[iteration] = out
    
    all_port = pd.DataFrame(all_port)
    all_port.T["sharpe"].plot(linewidth = 0, marker = '.')
    plt.show()