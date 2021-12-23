import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt

class PriceFetcher:
    def __init__(self):
        pass

    
    def fetch_info(self, ticker, start, end):
        
        start = dt.datetime(*start)
        end = dt.datetime(*end)
        #end = dt.datetime.now()
        info_fetched = web.DataReader(ticker, 'yahoo', start, end)
        return info_fetched
    

