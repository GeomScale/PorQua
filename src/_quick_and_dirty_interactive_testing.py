'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard

Licensed under GNU LGPL.3, see LICENCE file
'''


############################################################################
### QUICK-AND-DIRTY TESTING IN INTERACTIVE MODE DURING DEVELOPMENT
############################################################################





%reload_ext autoreload
%autoreload 2




import numpy as np
import pandas as pd

from backtest import Backtest, BacktestService
from optimization import (
    LeastSquares,
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
from helper_functions import load_data_msci




    


path_to_data = '../data/'  # Change this to your path

# Prepare data
data = load_data_msci(path = path_to_data, n = 24)
data


# Define rebalancing dates
n_days = 21 * 3
start_date = '2010-01-01'
dates = data['return_series'].index
rebdates = dates[dates > start_date][::n_days].strftime('%Y-%m-%d').tolist()
rebdates





    
# Flexibility over safety !



'''
Define the selection item builders.
SelectionItemBuilder is a callable class which takes a function (bibfn) as argument.
The function bibfn is a custom function that builds a selection item, i.e. a
pandas Series of boolean values indicating the selected assets at a given rebalancing date.
The function bibfn takes the backtest service (bs) and the rebalancing date (rebdate) as arguments.
Additional keyword arguments can be passed to bibfn using the arguments attribute of the SelectionItemBuilder instance.
The selection item is then added to the Selection attribute of the backtest service using the add_item method.
'''

selection_item_builders = {
    'data': SelectionItemBuilder(bibfn = bibfn_selection_data),
    # 'ltr': SelectionItemBuilder(bibfn = bibfn_selection_ltr),
    # 'growth': SelectionItemBuilder(bibfn = bibfn_selection_growth, sector_stdz = True),
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
    'return_series': OptimizationItemBuilder(bibfn = bibfn_return_series, width =365 * 3),
    'bm_series': OptimizationItemBuilder(bibfn = bibfn_bm_series, width = 365 * 3),
    'budget_constraint': OptimizationItemBuilder(bibfn = bibfn_budget_constraint, budget = 1),
    'box_constraints': OptimizationItemBuilder(bibfn = bibfn_box_constraints),
}

# Define the optimization
optimization = LeastSquares(
    solver_name = 'cvxopt',
)



# Initialize the backtest service
bs = BacktestService(
    data = data, 
    selection_item_builders = selection_item_builders,
    optimization_item_builders = optimization_item_builders,
    optimization = optimization,
    rebdates = rebdates,
)


# Instantiate the backtest
bt = Backtest(service = bs)

# Run backtest
bt.run()



bt.strategy.get_weights_df().plot()



bs.build_selection(rebdate = rebdates[0])
bs.build_optimization(rebdate = rebdates[0])
bs.selection.df()
bs.selection.selected






# Simulation
sim_bt = bt.strategy.simulate(return_series = bt.service.data['return_series'], fc = 0, vc = 0)


sim = pd.concat({
    'bm': bt.service.data['bm_series'],
    'sim': sim_bt,
}, axis = 1).dropna()

np.log((1 + sim)).cumsum().plot(figsize = (10, 6))


















