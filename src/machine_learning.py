# GeoFin : a python library for portfolio optimization and index replication
# GeoFin is part of GeomScale project

# Copyright (c) 2024 Cyril Bachelard
# Copyright (c) 2024 Minh Ha Ho

# Licensed under GNU LGPL.3, see LICENCE file



# %reload_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from optimization_data import OptimizationData


if __name__ == '__main__':
    path = '../data/'  # Change this to your path
    # Load data
    return_series = pd.read_parquet(f'{path}usa_returns.parquet')
    features = pd.read_parquet(f'{path}usa_features.parquet')

    # --------------------------------------------------------------------------
    # Regression
    # --------------------------------------------------------------------------

    # Prepare data
    series_id = return_series.columns[0]
    y = return_series[series_id]
    # Aggregating returns to monthly frequency
    y_monthly = np.exp(np.log(1 + y).resample('M').sum()) - 1

    X = features[features.index.get_level_values(1).isin([series_id])].reset_index(level=1, drop=True)
    X.fillna(0, inplace=True)

    training_data = OptimizationData(align = True, lags = {'y': -1}, X = X, y = y_monthly)

    # Using statsmodels

    # X = sm.add_constant(X)
    model = sm.OLS(training_data['y'], training_data['X'])
    results = model.fit()
    print(results.summary())

    # Assuming test_data is your new data
    predictions = results.predict(training_data['X'].tail(1))
    predictions, y_monthly.tail(1)


    # Using xgboost

    from xgboost import XGBRegressor

    # Instantiate the XGBRegressor
    xgb_model = XGBRegressor(objective ='reg:squarederror')

    # Fit the model to the training data
    xgb_model.fit(training_data['X'], training_data['y'])

    # Make predictions on the last row of the training data
    xgb_predictions = xgb_model.predict(training_data['X'].tail(1).values)

    xgb_predictions, y_monthly.tail(1)

    # Get feature importance
    feature_importance = xgb_model.get_booster().get_score(importance_type='weight')

    pd.Series(feature_importance).sort_values(ascending=False).head(10)


    # Compare the two models
    tmp = pd.concat({'beta': results.params,
                    'pvalues': results.pvalues,
                    'feat_imp': pd.Series(feature_importance)}, axis = 1)
    tmp.sort_values('feat_imp', ascending = False).head(10)
    tmp.sort_values('pvalues', ascending = True).head(10)
    tmp.sort_values('beta', ascending = False).head(10)


    tmp.plot.scatter(x = 'pvalues', y = 'feat_imp')
    plt.show()
    tmp.plot.scatter(x = 'beta', y = 'feat_imp')
    plt.show()


    # --------------------------------------------------------------------------
    # Learning to rank
    # --------------------------------------------------------------------------

    # Prepare the features
    X = features[features.index.get_level_values(1).isin([series_id])].reset_index(level=1, drop=True)
    X.fillna(0, inplace=True)

    # Prepare the labels
    ##  Aggregating returns to monthly frequency
    return_series_monthly = np.exp(np.log(1 + return_series).resample('M').sum()) - 1

    ##  Compute the ranks for each row of the monthly return series DataFrame
    ## We multiply the returns by -1 to ensure that the rank is in ascending order (i.e. the highest return is ranked 1)
    labels = (return_series_monthly * -1).rank(axis = 1)

    # Add the features and labels (ranks) to the training data. Lag the labels by one month.
    training_data = OptimizationData(align = True, lags = {'labels': -1}, X = X, labels = labels)
