# GeoFin : a python library for portfolio optimization and index replication
# GeoFin is part of GeomScale project

# Copyright (c) 2024 Cyril Bachelard
# Copyright (c) 2024 Minh Ha Ho

# Licensed under GNU LGPL.3, see LICENCE file



import numpy as np

from optimization import *
from constraints import Constraints
from optimization_data import OptimizationData
from portfolio import Portfolio, Strategy
from universe_selection import UniverseSelection
from typing import Callable, Dict, List


class Backtest:

    def __init__(self, rebdates: List[str], selection_model: UniverseSelection = None, **kwargs):
        self.rebdates = rebdates
        self.data = {}
        self.selection_model = selection_model
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
        self.optimization_data['X'] = X[X.index <= rebdate].tail(width+1)
        self.optimization_data['y'] = y[y.index <= rebdate].tail(width+1)

        return None

    def run(self) -> None:

        rebdates = self.rebdates
        for rebdate in rebdates:
            if not self.settings.get('quiet'):
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

            self.optimization.params['x0'] = x_init

            # select universe
            if self.selection_model is not None:
                universe = self.selection_model.select_universe()
                self.optimization.constraints.selection = universe

            ## Set objective
            self.optimization.set_objective(optimization_data = self.optimization_data)

            # Solve the optimization problem
            try:
                self.optimization.solve()
                weights = self.optimization.results['weights']
                portfolio = Portfolio(rebalancing_date = rebdate, weights = weights, init_weights = x_init)
                self.strategy.portfolios.append(portfolio)
            except Exception as error:
                raise RuntimeError(error)
            finally:
                # Append the optimized portfolio to the strategy, especially the failed ones for debugging
                self.models.append(self.optimization.model)

        return None

    def compute_summary(self, fc: float = 0, vc: float = 0):
        # Simulation
        sim_bt = self.strategy.simulate(return_series = self.data['return_series'], fc = fc, vc = vc)
        turnover = self.strategy.turnover(return_series = self.data['return_series'])

        # Analyze weights
        weights = self.strategy.get_weights_df()

        # Analyze simulation
        sim = pd.concat({'sim': sim_bt, 'index': self.data['return_series_index']}, axis = 1).dropna()
        sim.columns = sim.columns.get_level_values(0)
        sim = np.log(1 + sim).cumsum()

        return {'number_of_assets' : self.strategy.number_of_assets(),
                'returns' : sim,
                'turnover': turnover,
                'weights' : weights}
