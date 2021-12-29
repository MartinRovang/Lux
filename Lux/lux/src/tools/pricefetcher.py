import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
import time

def fetch_info(ticker, start, end):
    
    start_ = dt.datetime(*start)
    end_ = dt.datetime(*end)
    #end = dt.datetime.now()

    try:
        info_fetched = web.DataReader(ticker, 'yahoo', start_, end_)
        return info_fetched
    except:
        print("Error fetching data for {}".format(ticker))
        return False

