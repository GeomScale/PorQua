
############################################################################
### GSCO 2024 - BACKTEST
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     13.06.2024
# First version:    13.06.2024
# --------------------------------------------------------------------------


# Load base and 3rd party packages
import pandas as pd
import numpy as np
from optimization import *
from constraints import Constraints
from optimization_data import OptimizationData
from portfolio import Portfolio, Strategy
from typing import Callable, Dict, List


class Backtest:

    def __init__(self, rebdates: List[str], **kwargs):
        self.rebdates = rebdates
        self.data = {}
        self.strategy: Strategy = Strategy([])
        self.optimization: Optimization = None
        self.optimization_data: OptimizationData = OptimizationData(align = False)
        self.models = [] # for debug
        self.settings = {**kwargs}

    def prepare_optimization_data(self, rebdate: str) -> None:

        # Arguments
        width = self.settings.get('width')

        # Subset the return series
        X = self.data['return_series']
        y = self.data['return_series_index']
        if width is None:
            width = np.min(X.shape[0] - 1, y.shape[0] - 1)
        self.optimization_data['X'] = to_numpy(X[X.index <= rebdate].tail(width+1))
        self.optimization_data['y'] = to_numpy(y[y.index <= rebdate].tail(width+1))

        return None

    def run(self) -> None:

        rebdates = self.rebdates
        for rebdate in rebdates:
            print(f"Rebalancing date: {rebdate}")

            # Prepare optimization and solve it
            ## Prepare optimization data
            self.prepare_optimization_data(rebdate = rebdate)
            ## Load previous portfolio (i.e., the one from last rebalancing if such exists).
            prev_portfolio = self.strategy.get_initial_portfolio(rebalancing_date = rebdate)
            ## Floated weights
            x_init = prev_portfolio.initial_weights(selection = self.data['return_series'].columns,
                                                return_series = self.data['return_series'],
                                                end_date = rebdate,
                                                rescale = False)

            self.optimization.params['x_init'] = x_init

            ## Set objective
            self.optimization.set_objective(optimization_data = self.optimization_data)

            # Solve the optimization problem
            self.optimization.solve()

            # Append the optimized portfolio to the strategy
            weights = self.optimization.results['weights']
            self.models.append(self.optimization.model)

            portfolio = Portfolio(rebalancing_date = rebdate, weights = weights)
            self.strategy.portfolios.append(portfolio)

        return None
