
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
            portfolio = Portfolio(rebalancing_date = rebdate, weights = weights)
            self.strategy.portfolios.append(portfolio)

        return None







path = '../data/'  # Change this to your path



# Prepare data
return_series = pd.read_parquet(f'{path}usa_returns.parquet')
return_series_index = pd.read_csv(f'{path}SPTR.csv', index_col = 0)
return_series_index.index = pd.to_datetime(return_series_index.index, format='%d/%m/%Y')
features = pd.read_parquet(f'{path}usa_features.parquet')
data = {'return_series': return_series,
        'return_series_index': return_series_index,
        'features': features}

# Define rebalancing dates
n_days = 21
start_date = '2010-01-01'
dates = data['return_series'].index
rebdates = dates[dates > start_date][::n_days].strftime('%Y-%m-%d').tolist()

# Define constraints
constraints = Constraints(selection = data['return_series'].columns)
constraints.add_budget()
constraints.add_box(box_type = 'LongOnly')

# Define optimization
optimization = LeastSquares(solver_name = 'highs')
optimization.constraints = constraints

# Initialize backtest object
bt = Backtest(rebdates = rebdates,
              width = 252)
bt.data = data
bt.optimization = optimization

# Run backtest
bt.run()

# Simulation
sim_bt = bt.strategy.simulate(return_series = bt.data['return_series'],
                              fc = 0,
                              vc = 0)


# Analyze weights
bt.strategy.get_weights_df()
bt.strategy.number_of_assets().plot()


# Analyze simulation
sim = pd.concat({'sim': sim_bt,
                 'index': bt.data['return_series_index']}, axis = 1).dropna()
sim = np.log(1 + sim).cumsum().plot()


# Performance metrics
# ...













