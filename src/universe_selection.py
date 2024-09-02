# GeoFin : a python library for portfolio optimization and index replication
# GeoFin is part of GeomScale project

# Copyright (c) 2024 Cyril Bachelard
# Copyright (c) 2024 Minh Ha Ho

# Licensed under GNU LGPL.3, see LICENCE file

from abc import ABC, abstractmethod
import random
import pandas as pd
import tensorflow as tf


class UniverseSelection(ABC):
    def __init__(self,
                 model: str,
                 data: pd.DataFrame):
        self.model = model
        self.data = data

    @abstractmethod
    def build_model(self):
        return None

    @abstractmethod
    def select(self, stock_returns, nb_stocks=20):
        return None

    @abstractmethod
    def load_model(self, model_path: str = None):
        return None


class LstmSelection(UniverseSelection):

    def load_model(self, model_path: str = "../model/lstm_msci_00.keras"):
        self.trained_model = tf.keras.models.load_model(model_path)
        # Show the model architecture
        self.trained_model.summary()

    def select(self, stock_returns, nb_stocks=20):
        result_loaded_model = self.trained_model(stock_returns)
        indx_top = result_loaded_model[-1].numpy().argsort()[:nb_stocks]
        return self.data.columns[indx_top]  # Name/code of stocks
    def build_model(self):
        return None

class DummySelection(UniverseSelection):
    def select(self, stock_returns, nb_stocks=20):
        return random.sample(list(stock_returns.columns), nb_stocks)
    def load_model(self, model_path: str = None):
        return None
    def build_model(self):
        return None

#------------------- Helpers -------------------
def train_test_split(X, y=None, queryonnx_model_path=None, test_size=0.2):
    nb_test = int(test_size * len(X))
    test_index = X.index[-nb_test:]
    train_index = X.index[:X.shape[0] - nb_test]

    X_test = X.loc[test_index]
    X_train = X.loc[train_index]

    if y is not None:
        y_test = y.loc[test_index]
        y_train = y.loc[train_index]
    else:
        y_test, y_train = None, None

    if queryonnx_model_path is not None:
        query_test = queryonnx_model_path.loc[test_index]
        query_train = queryonnx_model_path.loc[train_index]
    else:
        query_test = None
        query_train = None
    return X_train, X_test, y_train, y_test, query_train, query_test
