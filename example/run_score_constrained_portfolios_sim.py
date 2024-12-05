# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root directory to Python path
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.append(project_root)
sys.path.append(src_path)

# Local application imports
from backtest import (
    Backtest,
    BacktestService,
    append_custom
)
from mean_estimation import MeanEstimator
from optimization import (
    PercentilePortfolios,
    ScoreConstrainedPortfolios,
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
    load_data_msci,
)
from helper_functions import output_to_strategies

path_to_data = '/home/quantagonia/PorQua/data/'

# Prepare data
data = load_data_msci(path = path_to_data, n = 24)

# Define rebalancing dates
n_days = 21 * 3
start_date = '2010-01-01'
dates = data['return_series'].index
rebdates = dates[dates > start_date][::n_days].strftime('%Y-%m-%d').tolist()
rebdates

selection_item_builders = {
    'data': SelectionItemBuilder(bibfn = bibfn_selection_data),
    # 'ltr': SelectionItemBuilder(bibfn = bibfn_selection_ltr),
    # 'volume': SelectionItemBuilder(bibfn = bibfn_selection_volume, sector_stdz = True),
}

optimization_item_builders = {
    'return_series': OptimizationItemBuilder(bibfn = bibfn_return_series, width = 365 * 3),
    'bm_series': OptimizationItemBuilder(bibfn = bibfn_bm_series, width = 365 * 3),
    'budget_constraint': OptimizationItemBuilder(bibfn = bibfn_budget_constraint, budget = 1),
    'box_constraints': OptimizationItemBuilder(bibfn = bibfn_box_constraints, upper = 0.1),
}

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
#bs.optimization = PercentilePortfolios(
#    n_percentile = 5,
#    estimator = mean_estimator,
#)
equivolume_distanced_mode = False
bs.optimization = ScoreConstrainedPortfolios(
    estimator = mean_estimator,
    n_score_levels = 5,
    equivolume_distanced = equivolume_distanced_mode,
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
#np.log((1 + sim_qp)).cumsum().plot(figsize = (10, 6))
plt.figure(figsize=(10, 6))
for col in sim_qp.columns:
    plt.plot(np.log(1 + sim_qp)[col].cumsum(), label=col)
plt.grid(True)
plt.legend()
plt.show()
