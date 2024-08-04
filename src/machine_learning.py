# GeoFin : a python library for portfolio optimization and index replication
# GeoFin is part of GeomScale project

# Copyright (c) 2024 Cyril Bachelard
# Copyright (c) 2024 Minh Ha Ho

# Licensed under GNU LGPL.3, see LICENCE file



# %reload_ext autoreload
# %autoreload 2

# Load base and 3rd party packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from optimization_data import OptimizationData
import onnx
from abc import ABC, abstractmethod
import tensorflow as tf

#### Calculate the metrics RMSE and MAPE ####
def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE)
    """
    rmse = np.sqrt(np.mean((np.array(y_true) - np.array(y_pred.values)) ** 2))
    return rmse


def calculate_mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) %
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def show_result(predictions, y_test, y_actual, method = None):
    print(f'RMSE of linear regression: {calculate_rmse(y_test, predictions)}')
    print(f'MAPE of linear regression: {calculate_mape(y_test, predictions)}')

    plt.plot(y_actual, color = 'cyan')
    plt.plot(predictions, color = 'green')
    plt.legend(["True values", "Prediction"])
    plt.title(method)
    plt.show()
class Universe_selection(ABC):
    def __init__(self,
                 model: str,
                 data: pd.DataFrame):
        self.model = model
        self.data = data

    @abstractmethod
    # def build_model(self):
    #     self.trained_model = None
    #     return None

    def load_model(self, model_path: str = None):
        self.trained_model = onnx.load(model_path)
        print(onnx.checker.check_model(self.trained_model))

    def select_universe(self, X, nb_stocks = 20):
        self.trained_model.predict(X)
        return None

class LSTM_revelance(Universe_selection):

    def load_model(self, model_path: str = "../model/lstm_msci_00.keras"):
        self.trained_model = tf.keras.models.load_model(model_path)
        # Show the model architecture
        self.trained_model .summary()

    def select_universe(self, X, nb_stocks = 20):
        result_loaded_model = self.trained_model(X)
        indx_top = result_loaded_model[-1].numpy().argsort()[:nb_stocks]
        return self.data.columns[indx_top] # Name/code of stocks

def train_test_split(X, y = None, queryonnx_model_path = None, test_size = 0.2) :
    nb_test = int(test_size * len(X))
    test_index = X.index[-nb_test:]
    train_index = X.index[:X.shape[0] - nb_test]

    X_test  = X.loc[test_index]
    X_train = X.loc[train_index]

    if not(y is None):
        y_test  = y.loc[test_index]
        y_train = y.loc[train_index]
    else:
        y_test, y_train = None, None

    if not(query is None):
        query_test = query.loc[test_index]
        query_train = query.loc[train_index]
    else:
        query_test = None
        query_train = None
    return X_train, X_test, y_train, y_test, query_train, query_test




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
