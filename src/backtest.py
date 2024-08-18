# GeoFin : a python library for portfolio optimization and index replication
# GeoFin is part of GeomScale project

# Copyright (c) 2024 Cyril Bachelard
# Copyright (c) 2024 Minh Ha Ho

# Licensed under GNU LGPL.3, see LICENCE file


from optimization import *
from constraints import Constraints
from optimization_data import OptimizationData
from portfolio import Portfolio, Strategy
from universe_selection import UniverseSelection
from typing import Dict, List


class Backtest:

    def __init__(self, rebdates: List[str], data: Dict, universe=None, selection_model: UniverseSelection = None, constraint_provider: 'BacktestConstraintProvider' = None, **kwargs):
        self.rebdates = rebdates
        self.data = data
        self.strategy: Strategy = Strategy([])
        self.universe = universe
        self.selection_model = selection_model
        self.constraint_provider: BacktestConstraintProvider = constraint_provider
        self.optimization: Optimization = None
        self.models = []  # for debug
        self.settings = {**kwargs}

    def prepare_optimization_data(self, rebdate: str) -> OptimizationData:
        # Subset the return series
        X = self.data['return_series']
        y = self.data['return_series_index']

        width = self.settings.get('width')
        if width is None:
            width = min(X.shape[0] - 1, y.shape[0] - 1)

        data = OptimizationData(align=False)
        data['X'] = X[X.index <= rebdate].tail(width + 1)
        data['y'] = y[y.index <= rebdate].tail(width + 1)

        return data

    def prepare_optimization(self, rebdate: str, x_init: dict) -> Optimization:
        optimization = self.optimization  # need to copy here
        optimization.params['x0'] = x_init

        # Prepare optimization data
        data = self.prepare_optimization_data(rebdate=rebdate)

        # select universe
        if self.constraint_provider is not None:
            universe = self.selection_model.select(data['X'], nb_stocks=20) if self.selection_model is not None else \
                        data['X'].columns.tolist()
            optimization.constraints = self.constraint_provider.build_constraints(universe)

        optimization.set_objective(optimization_data=data)
        return optimization

    def run(self) -> None:

        rebdates = self.rebdates
        for rebdate in rebdates:
            if not self.settings.get('quiet'):
                print(f"Rebalancing date: {rebdate}")

            # Load previous portfolio (i.e., the one from last rebalancing if exists).
            prev_portfolio = self.strategy.get_initial_portfolio(rebalancing_date=rebdate)
            # Floated weights
            x_init = prev_portfolio.initial_weights(selection=self.data['return_series'].columns,
                                                    return_series=self.data['return_series'],
                                                    end_date=rebdate,
                                                    rescale=False)

            # Prepare optimization data
            optimization = self.prepare_optimization(rebdate, x_init)

            # Solve the optimization problem
            try:
                optimization.solve()
                weights = optimization.results['weights']
                portfolio = Portfolio(rebalancing_date=rebdate, weights=weights, init_weights=x_init)
                self.strategy.portfolios.append(portfolio)
            except Exception as error:
                raise RuntimeError(error)
            finally:
                # Append the optimized portfolio to the strategy, especially the failed ones for debugging
                self.models.append(optimization.model)

        return None

    def compute_summary(self, fc: float = 0, vc: float = 0):
        # Simulation
        sim_bt = self.strategy.simulate(return_series=self.data['return_series'], fc=fc, vc=vc)
        turnover = self.strategy.turnover(return_series=self.data['return_series'])

        # Analyze weights
        weights = self.strategy.get_weights_df()

        # Analyze simulation
        sim = pd.concat({'sim': sim_bt, 'index': self.data['return_series_index']}, axis=1).dropna()
        sim.columns = sim.columns.get_level_values(0)
        sim = np.log(1 + sim).cumsum()

        return {'number_of_assets': self.strategy.number_of_assets(),
                'returns': sim,
                'turnover': turnover,
                'weights': weights}


class BacktestConstraintProvider:

    def __init__(self) -> None:
        self.budget = {'sense': None, 'rhs': None}
        self.box = {'box_type': 'NA', 'lower': None, 'upper': None}
        self.l1 = {}
        return None

    def add_budget(self, rhs=1, sense='=') -> None:
        if self.budget.get('rhs') is not None:
            print("Existing budget constraint is overwritten")
        self.budget = {'sense': sense, 'rhs': rhs}
        return None

    def add_box(self,
                box_type="LongOnly",
                lower=None,
                upper=None) -> None:
        if box_type == "Unbounded":
            lower = float("-inf") if lower is None else lower
            upper = float("inf") if upper is None else upper
        elif box_type == "LongShort":
            lower = -1 if lower is None else lower
            upper = 1 if upper is None else upper
        elif box_type == "LongOnly":
            upper = 1 if upper is None else upper
            if lower is None:
                lower = 0
            elif lower < 0:
                raise ValueError("Negative lower bounds for box_type 'LongOnly'")

        self.box = {'box_type': box_type, 'lower': lower, 'upper': upper}
        return None

    def add_l1(self, name: str, rhs: float, x0=None) -> None:
        self.l1[name] = {'name': name, 'rhs': rhs, 'x0': x0}
        return None

    def build_constraints(self, universe=None) -> Constraints:
        if universe is None:
            raise ValueError(f'Universe is required to build constraints')

        constraints = Constraints(selection=universe)

        budget = self.budget
        constraints.add_budget(budget['rhs'], budget['sense'])

        boxcon = self.box
        if boxcon is not None and boxcon['box_type'] != 'NA':
            constraints.add_box(box_type=boxcon['box_type'], lower=boxcon['lower'], upper=boxcon['upper'])

        for l1_con in self.l1.values():
            constraints.add_l1(name=l1_con['name'], rhs=l1_con['rhs'], x0=l1_con['x0'])

        return constraints
