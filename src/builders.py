'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard

Licensed under GNU LGPL.3, see LICENCE file
'''


############################################################################
### CLASS BacktestItemBuilde AND BACKTEST ITEM BUILDER FUNCTIONS
############################################################################


# Notice:
# The logic underlying the approach to build backtest items favours flexibility over safety !



from typing import Any

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod





# --------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------

class BacktestItemBuilder(ABC):

    def __init__(self, **kwargs):
        self._arguments = {}
        self._arguments.update(kwargs)

    @property
    def arguments(self) -> dict[str, Any]:
        return self._arguments

    @arguments.setter
    def arguments(self, value: dict[str, Any]) -> None:
        self._arguments = value

    @abstractmethod
    def __call__(self, service, rebdate: str) -> None:
        raise NotImplementedError("Method '__call__' must be implemented in derived class.")



class SelectionItemBuilder(BacktestItemBuilder):

    def __call__(self, bs, rebdate: str) -> None:

        '''
        Build selection item from a custom function.
        '''

        selection_item_builder_fn = self.arguments.get('bibfn')
        if selection_item_builder_fn is None or not callable(selection_item_builder_fn):
            raise ValueError('bibfn is not defined or not callable.')

        item_value = selection_item_builder_fn(bs = bs, rebdate = rebdate, **self.arguments)
        item_name = self.arguments.get('item_name')

        # Add selection item
        bs.selection.add_filtered(filter_name = item_name, value = item_value)
        return None



class OptimizationItemBuilder(BacktestItemBuilder):

    def __call__(self, bs, rebdate: str) -> None:

        '''
        Build optimization item from a custom function.
        '''

        optimization_item_builder_fn = self.arguments.get('bibfn')
        if optimization_item_builder_fn is None or not callable(optimization_item_builder_fn):
            raise ValueError('bibfn is not defined or not callable.')

        # Call the backtest item builder function. Notice that the function returns None,
        # it modifies the backtest service in place.
        optimization_item_builder_fn(bs = bs, rebdate = rebdate, **self.arguments)
        return None




# --------------------------------------------------------------------------
# Backtest item builder functions (bibfn) - Selection
# --------------------------------------------------------------------------

def bibfn_selection_min_volume(bs, rebdate: str, **kwargs) -> pd.Series:

    # Arguments
    width = kwargs.get('width', 365)
    agg_fn = kwargs.get('agg_fn', np.median)
    min_volume = kwargs.get('min_volume', 500_000)

    # Volume data
    X_vol = (
        bs.data.get_volume_series(end_date = rebdate, width = width)
        .fillna(0).apply(agg_fn, axis = 0)
    )

    # Filtering
    ids = [col for col in X_vol.columns if agg_fn(X_vol[col]) >= min_volume]

    # Output
    series = pd.Series(np.ones(len(ids)), index = ids, name = 'minimum_volume')
    bs.rebalancing.selection.add_filtered(filter_name = series.name,
                                            value = series)
    return None



def bibfn_selection_data(bs, rebdate: str, **kwargs) -> pd.Series:

    '''
    Backtest item builder function for defining the selection
    based on all available return series.
    '''

    data = bs.data.get('return_series')
    if data is None:
        raise ValueError('Return series data is missing.')

    return pd.Series(np.ones(data.shape[1], dtype = int), index = data.columns, name = 'binary')


def bibfn_selection_ltr(bs, rebdate: str, **kwargs) -> pd.DataFrame:

    '''
    Backtest item builder function for defining the selection
    based on a Learn-to-Rank model.
    '''

    # Arguments
    params_xgb = kwargs.get('params_xgb')

    # Selection
    ids = bs.selection.selected

    # Extract data
    merged_df = bs.data.get('merged_df').copy()
    df_train = merged_df[merged_df['DATE'] < rebdate]#.reset_index(drop = True)
    df_test = merged_df[merged_df['DATE'] == rebdate]#.reset_index(drop = True)
    df_test = df_test[ df_test['ID'].isin(selected) ]
    ids = df_test['ID'].to_list()

    # Training data
    X_train = df_train.drop(['DATE', 'ID', 'label', 'ret'], axis=1)
    y_train = df_train['label']
    grouped_train = df_train.groupby('DATE').size().to_numpy()
    dtrain = xgb.DMatrix(X_train, label = y_train)
    dtrain.set_group(grouped_train)

    # Evaluation data
    X_test = df_test.drop(['DATE', 'ID', 'label', 'ret'], axis=1)
    grouped_test = df_test.groupby('DATE').size().to_numpy()
    dtest = xgb.DMatrix(X_test)
    dtest.set_group(grouped_test)

    # Train and predict
    bst = xgb.train(params_xgb, dtrain, 100)
    scores = bst.predict(dtest) * (-1)

    # # Extract feature importance
    # f_importance = bst.get_score(importance_type='gain')

    return pd.DataFrame({'values': scores,
                         'binary': np.ones(len(scores), dtype = int),
                        }, index = scores.index)



# --------------------------------------------------------------------------
# Backtest item builder functions (bibfn) - Optimization data
# --------------------------------------------------------------------------

def bibfn_return_series(bs, rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for return series.
    Prepares an element of bs.optimization_data with
    single stock return series that are used for optimization.
    '''

    # Arguments
    width = kwargs.get('width')

    # Selection
    ids = bs.selection.selected

    # Data
    data = bs.data.get('return_series')
    if data is None:
        raise ValueError('Return series data is missing.')

    # Subset return series
    return_series = data[data.index <= rebdate].tail(width)[ids]

    # Remove weekends
    return_series = return_series[return_series.index.dayofweek < 5]

    # Output
    bs.optimization_data['return_series'] = return_series
    return None


def bibfn_bm_series(bs, rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for benchmark series.
    Prepares an element of bs.optimization_data with 
    the benchmark series that is be used for optimization.
    '''

    # Arguments
    width = kwargs.get('width')
    align = kwargs.get('align')

    # Data
    data = bs.data.get('bm_series')
    if data is None:
        raise ValueError('Benchmark return series data is missing.')

    # Subset the benchmark series
    bm_series = data[data.index <= rebdate].tail(width)

    # Remove weekends
    bm_series = bm_series[bm_series.index.dayofweek < 5]

    # Append the benchmark series to the optimization data
    bs.optimization_data['bm_series'] = bm_series

    # Align the benchmark series to the return series
    if align:
        bs.optimization_data.align_dates(
            variable_names = ['bm_series', 'return_series'],
            dropna = True
        )

    return None


# --------------------------------------------------------------------------
# Backtest item builder functions - Optimization constraints
# --------------------------------------------------------------------------

def bibfn_budget_constraint(bs, rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for setting the budget constraint.
    '''

    # Arguments
    budget = kwargs.get('budget', 1)

    # Add constraint
    bs.optimization.constraints.add_budget(rhs = budget, sense = '=')
    return None


def bibfn_box_constraints(bs, rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for setting the box constraints.
    '''

    # Arguments
    lower = kwargs.get('lower', 0)
    upper = kwargs.get('upper', 1)
    box_type = kwargs.get('box_type', 'LongOnly')

    # Constraints
    bs.optimization.constraints.add_box(box_type = box_type,
                                        lower = lower,
                                        upper = upper)
    return None
