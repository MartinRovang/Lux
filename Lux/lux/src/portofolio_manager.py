import pickle
from lux import fetch_info
import datetime as dt
import numpy as np
from rich.console import Console
from rich.table import Table

class Portofolio:
    def __init__(self):
        self.prtf = {'tickers': []}

    def add_ticker(self, ticker):
        if ticker not in self.prtf:
            self.prtf['tickers'].append(ticker)

    def load_data(self):
        self.portofolio = pickle.load(open('portofolio.pkl', 'rb'))
    
    def write_data(self):
        pickle.dump(self.prtf, open('portofolio.pkl', 'wb'))
    
    def get_stock_stats(self):
        end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))
        tmp_fetch = []
        console = Console()
        table = Table(title="Stocks")
        table.add_column("Ticker", style="cyan", no_wrap=True)
        table.add_column("Volume", style="cyan", no_wrap=True)
        table.add_column("Adj Close", style="cyan", no_wrap=True)
        for ticker in self.prtf:
            fetched = fetch_info(self.prtf[ticker], (end[0]-1,1,1), end)
            tmp_fetch.append(fetched[["Volume", "Adj Close"]])
            table.add_row(f"{self.prtf[ticker][0]}", f"{fetched['Volume'].values[-1][0]}", f"{fetched['Adj Close'].values[-1][0]}")

        console.print(table)


        

        
        

    

    

    

    

        


