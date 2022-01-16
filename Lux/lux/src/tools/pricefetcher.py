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


def logger(text):
    with open('/code/backgroundtasks/tools/database/logger.txt', 'a') as f:
        f.write(f'{text} |{dt.datetime.now().strftime("%Y-%m-%d-%M-%S") }\n')


def fetch_info(ticker, start, end):
    
    start_ = dt.datetime(*start)
    end_ = dt.datetime(*end)
    #end = dt.datetime.now()

    try:
        info_fetched = web.DataReader(ticker, 'yahoo', start_, end_)
        return info_fetched
    except Exception as e:
        logger(f'{ticker} {e}')
        return False


def make_portofolios():
    while True:
        logger('Using make_portofolios function')
        time.sleep(50)
        end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
        if os.path.exists(f'/code/backgroundtasks/tools/database/stockprices/all_stock_values_{end}.pkl'):
            logger('Making portofolios')
            all_data_prices = pickle.load(open(f'/code/backgroundtasks/tools/database/stockprices/all_stock_values_{end}.pkl', 'rb'))
            all_data_prices_returns = all_data_prices.pct_change()
            portofolio_size = 10
            output = portofolio_kit.iter_portofolio(portofolio_size, all_data_prices_returns, N_iterations = 100_000)
            if os.path.exists(f'/code/backgroundtasks/tools/database/stockprices/sharpes_ports_{end}.pkl'):
                data  = pickle.load(open(f'/code/backgroundtasks/tools/databasestockprices/sharpes_ports_{end}.pkl', 'rb'))
                data = data.update(output)
                output = {k: v for k, v in sorted(output.items(), key=lambda item: item[1]['sharpe'], reverse=True)}
                output = dict(itertools.islice(output.items(), 200000)) 
                pickle.dump(output, open(f'/code/backgroundtasks/tools/database/stockprices/sharpes_ports_{end}.pkl', 'wb'))
            else:
                output = {k: v for k, v in sorted(output.items(), key=lambda item: item[1]['sharpe'], reverse=True)}
                pickle.dump(output, open(f'/code/backgroundtasks/tools/database/stockprices/sharpes_ports_{end}.pkl', 'wb'))




def grab_newest_data_auto():

    while True:
        logger('grab_newest_data_auto function')
        end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
        files = glob.glob('/code/backgroundtasks/tools/database/stockprices/*')
        if not os.path.exists(f'/code/backgroundtasks/tools/database/stockprices/all_stock_values_{end}.pkl'):
            logger('Grabbing data')
            for f in files:
                os.remove(f)
            all_stocks_OSLO = pd.read_csv('/code/backgroundtasks/tools/database/all_stocks_OSLO1.csv', encoding='latin', header=0, delimiter=';')
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
            pickle.dump(all_data_prices, open(f'/code/backgroundtasks/tools/database/stockprices/all_stock_values_{end}.pkl', 'wb'))
        
        if not os.path.exists(f'/code/backgroundtasks/tools/database/stockprices/sharpes_ports_{end}.pkl'):
            logger('Making sharpe portofolios')
            all_data_prices = pickle.load(open(f'/code/backgroundtasks/tools/database/stockprices/all_stock_values_{end}.pkl', 'rb'))
            all_data_prices_returns = all_data_prices.pct_change()
            portofolio_size = 10
            output = portofolio_kit.iter_portofolio(portofolio_size, all_data_prices_returns, N_iterations = 200_000)
            pickle.dump(output, open(f'/code/backgroundtasks/tools/database/stockprices/sharpes_ports_{end}.pkl', 'wb'))
            logger('End of making sharpe portofolios')
        else:
            time.sleep(2)