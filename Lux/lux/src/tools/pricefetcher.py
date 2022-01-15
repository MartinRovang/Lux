import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
import pickle
import os
import glob
import portofolio_kit
import itertools
import time

def fetch_info(ticker, start, end):
    
    start_ = dt.datetime(*start)
    end_ = dt.datetime(*end)
    #end = dt.datetime.now()

    try:
        info_fetched = web.DataReader(ticker, 'yahoo', start_, end_)
        return info_fetched
    except Exception as e:
        with open('error_log.txt', 'a') as f:
            f.write(f'{ticker} {e}. {dt.datetime.now().strftime("%Y-%m-%d-%m") }\n')
        return False



def make_portofolios():
    while True:
        time.sleep(50)
        end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
        if os.path.exists(f'lux/database/stockprices/all_stock_values_{end}.pkl'):
            with open('logger.txt', 'a') as f:
                f.write(f'making portofolio: {dt.datetime.now().strftime("%Y-%m-%d-%m") }\n')
            all_data_prices = pickle.load(open(f'lux/database/stockprices/all_stock_values_{end}.pkl', 'rb'))
            all_data_prices_returns = all_data_prices.pct_change()
            portofolio_size = 10
            output = portofolio_kit.iter_portofolio(portofolio_size, all_data_prices_returns, N_iterations = 100_000)
            if os.path.exists(f'lux/database/stockprices/sharpes_ports_{end}.pkl'):
                data  = pickle.load(open(f'lux/database/stockprices/sharpes_ports_{end}.pkl', 'rb'))
                data = data.update(output)
                output = {k: v for k, v in sorted(output.items(), key=lambda item: item[1]['sharpe'], reverse=True)}
                output = dict(itertools.islice(output.items(), 300000)) 
                pickle.dump(output, open(f'lux/database/stockprices/sharpes_ports_{end}.pkl', 'wb'))
            else:
                output = {k: v for k, v in sorted(output.items(), key=lambda item: item[1]['sharpe'], reverse=True)}
                pickle.dump(output, open(f'lux/database/stockprices/sharpes_ports_{end}.pkl', 'wb'))




def grab_newest_data_auto():

    while True:
        end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
        files = glob.glob('lux/database/stockprices/*')
        if not os.path.exists(f'lux/database/stockprices/all_stock_values_{end}.pkl'):
            with open('logger.txt', 'a') as f:
                f.write(f'grabbing data: {dt.datetime.now().strftime("%Y-%m-%d-%m") }\n')
            for f in files:
                os.remove(f)
            all_stocks_OSLO = pd.read_csv('lux/database/all_stocks_OSLO1.csv', encoding='latin', header=0, delimiter=';')
            all_stocks_OSLO['Symbol'] = all_stocks_OSLO['Symbol'] + '.OL'
            all_stocks_OSLO_symbols = all_stocks_OSLO['Symbol'].dropna()
            symbols_according_to_index = []
            all_data_prices = pd.DataFrame()
            for _ , stocksymbol in enumerate(all_stocks_OSLO_symbols):
                data = fetch_info(stocksymbol, (end[0]-5,end[1],end[2]), end)
                if type(data) == bool:
                    continue
                data_prices = data['Adj Close'].dropna()
                symbols_according_to_index.append(stocksymbol)
                all_data_prices = pd.concat([all_data_prices, data_prices], axis = 1)

            all_data_prices = pd.DataFrame(all_data_prices.values, index = all_data_prices.index, columns = symbols_according_to_index)
            pickle.dump(all_data_prices, open(f'lux/database/stockprices/all_stock_values_{end}.pkl', 'wb'))
        
        # if not os.path.exists(f'lux/database/stockprices/sharpes_ports_{end}.pkl'):
        #     all_data_prices = pickle.load(open(f'lux/database/stockprices/all_stock_values_{end}.pkl', 'rb'))
        #     all_data_prices_returns = all_data_prices.pct_change()
        #     portofolio_size = 10
        #     output = lux.iter_portofolio(portofolio_size, all_data_prices_returns, N_iterations = 100_000)
        #     pickle.dump(output, open(f'lux/database/stockprices/sharpes_ports_{end}.pkl', 'wb'))
        else:
            time.sleep(100)