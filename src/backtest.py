
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


class Backtest:

    def __init__(self, **kwargs):
        self.data = {}
        self.strategy: Strategy = Strategy([])
        self.optimization: Optimization = None
        self.optimization_data: OptimizationData = OptimizationData(align = False)
        self.summary = []
        self.settings = {**kwargs}

    def prepare_optimization_data(self, rebdate: str) -> None:

        # Arguments
        width = self.settings.get('width')

        # Subset the return series
        X = self.data['return_series']
        y = self.data['return_series_index']
        if width is None:
            width = np.min(X.shape[0] - 1, y.shape[0] - 1)
        self.optimization_data['X'] = X[X.index <= rebdate].tail(width+1).to_numpy()
        self.optimization_data['y'] = y[y.index <= rebdate].tail(width+1).to_numpy()

        return None

    def run(self) -> None:

        rebdates = self.settings['rebdates']
        for rebdate in rebdates:
            print(f"Rebalancing date: {rebdate}")

            # Prepare optimization and solve it
            ## Prepare optimization data
            self.prepare_optimization_data(rebdate = rebdate)
            ## Load initial portfolio (i.e., the one from last rebalancing if such exists).
            initial_portfolio = self.strategy.get_initial_portfolio(rebalancing_date = rebdate)
            ## Floated weights
            x_init = initial_portfolio.initial_weights(return_series = self.data['return_series'],
                                                       selection = self.data['return_series'].columns,
                                                       end_date = rebdate,
                                                       rescale = False)
            self.optimization.params['x_init'] = x_init

            ## Set objective
            self.optimization.set_objective(optimization_data = self.optimization_data)

            # Solve the optimization problem
            self.optimization.solve()

            # Append the optimized portfolio to the strategy
            weights = self.optimization.results['weights']

            self.summary.append(self.optimization.model)

            portfolio = Portfolio(rebalancing_date = rebdate, weights = weights)
            self.strategy.portfolios.append(portfolio)

        return None
