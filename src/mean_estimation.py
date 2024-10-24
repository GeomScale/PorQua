'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''


############################################################################
### MEAN ESTIMATOR
############################################################################


import pandas as pd
import numpy as np




class MeanEstimator():

    def __init__(self, **kwargs) -> None:
        self.spec = {
            'method': 'geometric',
            'scalefactor': 1,
            'n_mom': None,
            'n_rev': None
        }
        self.spec.update(kwargs)

    def estimate(self, X: pd.DataFrame) -> pd.DataFrame or pd.Series:
        fun = getattr(self, f'estimate_{self.spec["method"]}')
        mu = fun(X = X)
        return mu

    def estimate_geometric(self, X: pd.DataFrame):
        n_mom = X.shape[0] if self.spec.get('n_mom') is None else self.spec.get('n_mom')
        n_rev = 0 if self.spec.get('n_rev') is None else self.spec.get('n_rev')
        scalefactor = 1 if self.spec.get('scalefactor') is None else self.spec.get('scalefactor')
        X = X.tail(n_mom).head(n_mom-n_rev)
        mu = np.exp(np.log(1 + X).mean(axis=0) * scalefactor) - 1
        # Alternatively:
        # from scipy.stats import gmean
        # mu = (gmean(1 + X) - 1).tolist()
        return mu
