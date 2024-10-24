'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''


############################################################################
### CLASSES BacktestData, BacktestService, Backtest
############################################################################




import os
from typing import Optional
import pickle

from optimization import Optimization, EmptyOptimization
from optimization_data import OptimizationData
from constraints import Constraints
from portfolio import Portfolio, Strategy
from selection import Selection
from builders import SelectionItemBuilder, OptimizationItemBuilder





class BacktestData():

    def __init__(self):
        pass


class BacktestService():

    def __init__(self,
                 data: BacktestData,
                 selection_item_builders: dict[str, SelectionItemBuilder],
                 optimization_item_builders: dict[str, OptimizationItemBuilder],
                 optimization: Optional[Optimization] = EmptyOptimization(),
                 settings: Optional[dict] = None,
                 **kwargs) -> None:
        self.data = data
        self.optimization = optimization
        self.selection_item_builders = selection_item_builders
        self.optimization_item_builders = optimization_item_builders
        self.settings = settings if settings is not None else {}
        self.settings.update(kwargs)
        # Initialize the selection and optimization data
        self.selection = Selection()
        self.optimization_data = OptimizationData([])

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(self, value):
        if not isinstance(value, Selection):
            raise TypeError("Expected a Selection instance for 'selection'")
        self._selection = value

    @property
    def selection_item_builders(self):
        return self._selection_item_builders

    @selection_item_builders.setter
    def selection_item_builders(self, value):
        if not isinstance(value, dict) or not all(
            isinstance(v, SelectionItemBuilder) for v in value.values()
        ):
            raise TypeError(
                "Expected a dictionary containing SelectionItemBuilder instances "
                "for 'selection_item_builders'"
            )
        self._selection_item_builders = value

    @property
    def optimization(self):
        return self._optimization

    @optimization.setter
    def optimization(self, value):
        if not isinstance(value, Optimization):
            raise TypeError("Expected an Optimization instance for 'optimization'")
        self._optimization = value

    @property
    def optimization_item_builders(self):
        return self._optimization_item_builders

    @optimization_item_builders.setter
    def optimization_item_builders(self, value):
        if not isinstance(value, dict) or not all(
            isinstance(v, OptimizationItemBuilder) for v in value.values()
        ):
            raise TypeError(
                "Expected a dictionary containing OptimizationItemBuilder instances "
                "for 'optimization_item_builders'"
            )
        self._optimization_item_builders = value

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, value):
        if not isinstance(value, dict):
            raise TypeError("Expected a dictionary for 'settings'")
        self._settings = value

    def build_selection(self, rebdate: str) -> None:
        # Loop over the selection_item_builders items
        for key, item_builder in self.selection_item_builders.items():
            item_builder.arguments['item_name'] = key
            item_builder(self, rebdate)
        return None

    def build_optimization(self, rebdate: str) -> None:

        # Initialize the optimization constraints
        self.optimization.constraints = Constraints(selection = self.selection.selected)

        # Loop over the optimization_item_builders items
        for item_builder in self.optimization_item_builders.values():
            item_builder(self, rebdate)
        return None

    def prepare_rebalancing(self, rebalancing_date: str) -> None:
        self.build_selection(rebdate = rebalancing_date)
        self.build_optimization(rebdate = rebalancing_date)
        return None



class Backtest:

    def __init__(self) -> None:
        self._strategy = Strategy([])
        self._output = {}

    @property
    def strategy(self):
        return self._strategy

    @property
    def output(self):
        return self._output

    def append_output(self,
                      date_key = None,
                      output_key = None,
                      value = None):
        if value is None:
            return True

        if date_key in self.output.keys():
            if output_key in self.output[date_key].keys():
                raise Warning(f"Output key '{output_key}' for date key '{date_key}' \
                    already exists and will be overwritten.")
            self.output[date_key][output_key] = value
        else:
            self.output[date_key] = {}
            self.output[date_key].update({output_key: value})

        return True

    def rebalance(self,
                  bs: BacktestService,
                  rebalancing_date: str) -> None:

        # Prepare the rebalancing, i.e., the optimization problem
        bs.prepare_rebalancing(rebalancing_date = rebalancing_date)

        # Solve the optimization problem
        try:
            bs.optimization.set_objective(optimization_data = bs.optimization_data)
            bs.optimization.solve()
        except Exception as error:
            raise RuntimeError(error)

        return None

    def run(self, bs: BacktestService) -> None:

        for rebalancing_date in bs.settings['rebdates']:

            if not bs.settings.get('quiet'):
                print(f'Rebalancing date: {rebalancing_date}')

            self.rebalance(bs = bs,
                           rebalancing_date = rebalancing_date)

            # Append portfolio to strategy
            weights = bs.optimization.results['weights']
            portfolio = Portfolio(rebalancing_date = rebalancing_date, weights = weights)
            self.strategy.portfolios.append(portfolio)

            # Append stuff to output if a custom append function is provided
            append_fun = bs.settings.get('append_fun')
            if append_fun is not None:
                append_fun(
                    backtest = self,
                    bs = bs,
                    rebdate = rebalancing_date,
                    what = bs.settings.get('append_fun_args')
                )

        return None

    def save(self,
             filename: str,
             path: Optional[str] = None) -> None:
        try:
            if path is not None and filename is not None:
                filename = os.path.join(path, filename)   #// alternatively, use pathlib package
            with open(filename, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object:", ex)

        return None



# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def append_custom(backtest: Backtest,
                  bs: BacktestService,
                  rebalancing_date: Optional[str] = None,
                  what: Optional[list] = None) -> None:

    if what is None:
        what = ['w_dict', 'objective']

    for key in what:
        if key == 'w_dict':
            w_dict = bs.optimization.results['w_dict']
            for key in w_dict.keys():
                weights = w_dict[key]                    
                if hasattr(weights, 'to_dict'):
                    weights = weights.to_dict()
                portfolio = Portfolio(rebalancing_date = rebalancing_date, weights = weights)
                backtest.append_output(date_key = rebalancing_date,
                                        output_key = f'weights_{key}',
                                        value = pd.Series(portfolio.weights))
        else:
            if not key in bs.optimization.results.keys():
                continue
            backtest.append_output(date_key = rebalancing_date,
                                    output_key = key,
                                    value = bs.optimization.results[key])
    return None
