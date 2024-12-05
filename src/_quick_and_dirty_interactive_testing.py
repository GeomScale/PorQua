'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard

Licensed under GNU LGPL.3, see LICENCE file
'''


############################################################################
### QUICK-AND-DIRTY TESTING IN INTERACTIVE MODE DURING DEVELOPMENT
############################################################################






# %reload_ext autoreload
# %autoreload 2

# Third party imports
import numpy as np
import pandas as pd

# Local application imports
from backtest import (
    Backtest,
    BacktestService,
    append_custom
)
from mean_estimation import MeanEstimator
from optimization import (
    LeastSquares,
    WeightedLeastSquares,
    QEQW,
    LAD,
    PercentilePortfolios,
)
from builders import (
    SelectionItemBuilder,
    OptimizationItemBuilder,
    bibfn_selection_data,
    bibfn_return_series,
    bibfn_bm_series,
    bibfn_budget_constraint,
    bibfn_box_constraints,
)
from data_loader import (
    load_pickle,
    load_data_msci,
)
from helper_functions import output_to_strategies






# --------------------------------------------------------------------------
# Load data and prepare backtest service
# --------------------------------------------------------------------------



path_to_data = '../data/'

# Prepare data
data = load_data_msci(path = path_to_data, n = 24)
data


# Define rebalancing dates
n_days = 21 * 3
start_date = '2010-01-01'
dates = data['return_series'].index
rebdates = dates[dates > start_date][::n_days].strftime('%Y-%m-%d').tolist()
rebdates




'''
Define the selection item builders.
SelectionItemBuilder is a callable class which takes a function (bibfn) as argument.
The function bibfn is a custom function that builds a selection item, i.e. a
pandas Series of boolean values indicating the selected assets at a given rebalancing date.
The function bibfn takes the backtest service (bs) and the rebalancing date (rebdate) as arguments.
Additional keyword arguments can be passed to bibfn using the arguments attribute of the SelectionItemBuilder instance.
The selection item is then added to the Selection attribute of the backtest service using the add_item method.
To inspect the current instance of the selection object, type bs.selection.df()
'''

selection_item_builders = {
    'data': SelectionItemBuilder(bibfn = bibfn_selection_data),
    # 'ltr': SelectionItemBuilder(bibfn = bibfn_selection_ltr),
    # 'volume': SelectionItemBuilder(bibfn = bibfn_selection_volume, sector_stdz = True),
}






'''
Define the optimization item builders.
OptimizationItemBuilder is a callable class which takes a function (bibfn) as argument.
The function bibfn is a custom function that builds an item which is used for the optimization.
Such items can be constraints, which are added to the constraints attribute of the optimization object,
or datasets which are added to the instance of the OptimizationData class.
The function bibfn takes the backtest service (bs) and the rebalancing date (rebdate) as arguments.
Additional keyword arguments can be passed to bibfn using the arguments attribute of the OptimizationItemBuilder instance.
'''

optimization_item_builders = {
    'return_series': OptimizationItemBuilder(bibfn = bibfn_return_series, width = 365 * 3),
    'bm_series': OptimizationItemBuilder(bibfn = bibfn_bm_series, width = 365 * 3),
    'budget_constraint': OptimizationItemBuilder(bibfn = bibfn_budget_constraint, budget = 1),
    'box_constraints': OptimizationItemBuilder(bibfn = bibfn_box_constraints, upper = 0.1),
}



# Initialize the backtest service
bs = BacktestService(
    data = data,
    selection_item_builders = selection_item_builders,
    optimization_item_builders = optimization_item_builders,
    rebdates = rebdates,
)







# --------------------------------------------------------------------------
# Run backtests
# --------------------------------------------------------------------------


# Run backtest of a Quasi Equal Weights model
# Update the backtest service with the optimization object
bs.optimization = QEQW()
# Instantiate the backtest object and run the backtest
bt_qeqw = Backtest()
bt_qeqw.run(bs = bs)


# Run backtest of a Least Squares model
bs.optimization = LeastSquares(solver_name = 'cvxopt')
bt_ls = Backtest()
bt_ls.run(bs = bs)


# Run backtest of a Weighted Least Squares model
bs.optimization = WeightedLeastSquares(
    solver_name = 'cvxopt',
    tau = 252
)
bt_wls = Backtest()
bt_wls.run(bs = bs)


# Run backtest of a Least Absolute Deviation model
bs.optimization = LAD(solver_name = 'cvxopt')
bt_lad = Backtest()
bt_lad.run(bs = bs)





# # Save the backtests locally as pickle files
# save_path = ''
# bt_qeqw.save(path = save_path, filename = 'qeqw')
# bt_ls.save(path = save_path, filename = 'ls')
# bt_wls.save(path = save_path, filename = 'wls')
# bt_lad.save(path = save_path, filename = 'lad')


# # Load locally saved backtests from pickle files
# bt_qeqw = load_pickle(path = save_path, filename = 'qeqw')
# bt_ls = load_pickle(path = save_path, filename = 'ls')
# bt_wls = load_pickle(path = save_path, filename = 'wls')
# bt_lad = load_pickle(path = save_path, filename = 'lad')








# --------------------------------------------------------------------------
# Simulate strategies
# --------------------------------------------------------------------------

fixed_costs = 0
variable_costs = 0

sim_qeqw = bt_qeqw.strategy.simulate(return_series = bs.data['return_series'], fc = fixed_costs, vc = variable_costs)
sim_ls = bt_ls.strategy.simulate(return_series = bs.data['return_series'], fc = fixed_costs, vc = variable_costs)
sim_wls = bt_wls.strategy.simulate(return_series = bs.data['return_series'], fc = fixed_costs, vc = variable_costs)
# sim_lad = bt_lad.strategy.simulate(return_series = bs.data['return_series'], fc = fixed_costs, vc = variable_costs)

sim = pd.concat({
    'bm': bs.data['bm_series'],
    'qeqw': sim_qeqw,
    'ls': sim_ls,
    'wls': sim_wls,
#     'lad': sim_lad,
}, axis = 1).dropna()


np.log((1 + sim)).cumsum().plot(figsize = (10, 6))







# --------------------------------------------------------------------------
# Backtest quintile portfolios based on geometric mean estimator
# --------------------------------------------------------------------------

# Initialize the backtest service
bs = BacktestService(
    data = data,
    selection_item_builders = selection_item_builders,
    optimization_item_builders = optimization_item_builders,
    rebdates = rebdates,
    append_fun = append_custom,
    append_fun_args = ['w_dict']
)

# Define the mean estimator
mean_estimator = MeanEstimator(
    method = 'geometric',
    scalefactor = 1,
    n_mom = 252,
    n_rev = 21,
)

# Define portfolio optimization object and run backtest
bs.optimization = PercentilePortfolios(
    n_percentile = 5,
    estimator = mean_estimator,
)
bt_qp = Backtest()
bt_qp.run(bs = bs)


# Simulate the strategies
strat_dict = output_to_strategies(bt_qp.output)

# Simulate
fixed_costs = 0
variable_costs = 0
sim_dict = {
    key: value.simulate(return_series = bs.data['return_series'], fc = fixed_costs, vc = variable_costs)
    for key, value in strat_dict.items()
}
sim_qp = pd.concat(sim_dict, axis = 1).dropna()

# Plot
np.log((1 + sim_qp)).cumsum().plot(figsize = (10, 6))










