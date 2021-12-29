import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt

def fetch_info(ticker, start, end):
    
    start = dt.datetime(*start)
    end = dt.datetime(*end)
    #end = dt.datetime.now()
    try:
        info_fetched = web.DataReader(ticker, 'yahoo', start, end)
        return info_fetched
    except:
        return False

