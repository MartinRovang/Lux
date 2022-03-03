import numpy as np
import pandas as pd
import pickle
from pandas.core.frame import DataFrame

def portofolio_metrics(portofolio:DataFrame, weights:list = [], tradedays:int = 250, optimweights:bool = True):
        """
        Args:
            portofolio ([type]): [description]
            weights (list, optional): [description]. Defaults to [].
            optimweights (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """        """"""
        # https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf
        # SIGMA = np.cov(portofolio)

        assert tradedays == len(portofolio.index), 'tradedays must be equal to the length of the index'

        SIGMA = portofolio.cov()
        MU = portofolio.mean()
        # MU = np.mean(portofolio, axis = 1)*self.tradedays
        if optimweights == True:
            weights = np.ones(len(MU))
            SIGMA_INV = np.linalg.inv(SIGMA)
            weights = SIGMA_INV@weights/(weights.T@SIGMA_INV@weights) # Eq. (1.12) minimum variance portfolio weights


        mean = weights.T@MU
        mean = mean*tradedays
        std = np.sqrt(weights.T@SIGMA@weights)*np.sqrt(tradedays)

        # market_mean = np.mean(benchmark)*self.tradedays
        # market_var = np.var(benchmark)*self.tradedays
        # beta = std**2/market_var
        # alpha = mean/market_mean


        output = {'mean': round(mean,5), 'std': round(std,5), 'sharpe': round(mean/std,5), 'weights': [round(x,3) for x in weights]}
        return output



def iter_portofolio(portofolio_size, all_stock, N_iterations, tradedays = 250):
        """[summary]

        Args:
            portofolio_size ([type]): [description]
            all_stock ([type]): [description]
            benchmark ([type]): [description]
            N ([type]): [description]

        Returns:
            [type]: [description]
        """
        # https://quant.stackexchange.com/questions/53992/is-this-methodology-for-finding-the-minimum-variance-portfolio-with-no-short-sel


        print('Started iterating...')
        output = {}
        for _ in list(range(N_iterations)):
            current_portofolio = all_stock.sample(n = portofolio_size, replace = False, axis = 1)
            current_portofolio = current_portofolio.dropna()
            if len(current_portofolio.index) > 250:
                current_portofolio = current_portofolio.iloc[-250:]
                try:
                    result = portofolio_metrics(current_portofolio, tradedays = tradedays)
                    result['portofolio'] = current_portofolio.columns.values
                    output[result['sharpe']] = result
                except Exception as e:
                    print(e)
                    print('Error with: ', current_portofolio.columns.values)
        
        # sort dictionary by sharpe ratio
        output = {k: v for k, v in sorted(output.items(), key=lambda item: item[1]['sharpe'], reverse=True)}
        print('Finished iterating')
        return output