__version__ = '0.1.0'
from fastapi import FastAPI
import asyncio
import time
import threading
from lux.src.tools.pricefetcher import grab_newest_data_auto
from lux.src.tools.portofolio_kit import iter_portofolio
import pickle
import datetime as dt
import numpy as np
import json
import os
app = FastAPI()

x = threading.Thread(target = grab_newest_data_auto)
x.daemon = True
x.start()


@app.get("/")
async def read_root():
    return {'hello': 'world'}



@app.get("/portofolios/")
async def portofolio(filter: str = "weight gt -0.5 and std lt 0.5 and tot = 250"):
    # http://127.0.0.1:8000/portofolios/?filter=weight gt -0.5 and std lt 0.5
    filter_types = filter.split(" and ")
    filter_operations = [x.split(" ") for x in filter_types]

    end = tuple(np.array(dt.datetime.now().strftime('%Y-%m-%d').split('-')).astype('int'))

    if os.path.exists(f'lux/database/stockprices/sharpes_ports_{end}.pkl'):
        data  = pickle.load(open(f'lux/database/stockprices/sharpes_ports_{end}.pkl', 'rb'))
        best_ports = [x for x in data]
        output = []
        for port in best_ports:
            tempweights = np.array(data[port]['weights'])
            tempstds = np.array(data[port]['std'])
            temptot = 250

            for filterop in filter_operations:
                if filterop[0] == "weight":
                    if filterop[1] == "gt":
                        tempweights = tempweights > float(filterop[2])
                    elif filterop[1] == "lt":
                        tempweights = tempweights < float(filterop[2])

                elif filterop[0] == "std":
                    if filterop[1] == "gt":
                        tempstds = tempstds > float(filterop[2])
                    elif filterop[1] == "lt":
                        tempstds = tempstds < float(filterop[2])
                
                elif filterop[0] == "tot":
                    if filterop[1] == "=":
                        temptot = int(filterop[2])

                if all(tempweights) and tempstds:
                    output.append({'sharpeRatio':data[port]['sharpe'] , 'meanReturn':data[port]['mean'], 'std':data[port]['std'],'portofolio': list(data[port]['portofolio']), "weights":list(data[port]['weights'])})
            
            if len(output) > temptot:
                break
        return output
    else:
        return "Updating portofolios, come back later..."


