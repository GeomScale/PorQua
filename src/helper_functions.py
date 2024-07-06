# GeoFin : a python library for portfolio optimization and index replication
# GeoFin is part of GeomScale project

# Copyright (c) 2024 Cyril Bachelard
# Copyright (c) 2024 Minh Ha Ho

# Licensed under GNU LGPL.3, see LICENCE file



import os
from typing import Dict
import numpy as np
import pandas as pd
import pickle

def load_data(universe):
    if universe == 'msci':
        data = load_data_msci()
    elif universe == 'usa':
        data = load_data_usa()
    else:
        raise ValueError('Universe not recognized.')
    return data

def load_data_msci(path: str = None, n: int = 24) -> Dict[str, pd.DataFrame]:

    path = os.path.join(os.getcwd(), 'data/') if path is None else path
    # Load msci country index return series
    df = pd.read_csv(f'{path}msci_country_indices.csv',
                    sep=';',
                    index_col=0,
                    header=0,
                    parse_dates=True)
    df.index = pd.to_datetime(df.index, format='%d/%m/%Y')
    series_id = df.columns[0:n]
    X = df[series_id]

    # Load msci world index return series
    y = pd.read_csv(f'{path}NDDLWI.csv',
            sep=';',
            index_col=0,
            header=0,
            parse_dates=True)

    y.index = pd.to_datetime(y.index, format='%d/%m/%Y')

    data = {'X': X, 'y': y}
    return data


def load_data_usa(path: str = None) -> Dict[str, pd.DataFrame]:

    # Load U.S. security data
    path = os.path.join(os.getcwd(), 'data/') if not path else path
    df_secd = pd.read_csv(f'{path}usa_returns.csv', index_col = 0, parse_dates=True)
    df_secd.index = pd.to_datetime(df_secd.index, format='%Y-%m-%d')

    # Load U.S. stock characteristics (fundamentals) data
    # ...
    df_funda = None

    # Load S&P 500 index return series
    y = pd.read_csv(f'{path}SPTR.csv',
            index_col=0,
            header=0,
            parse_dates=True,
            dayfirst=True)

    y.index = pd.to_datetime(y.index, format='%d/%m/%Y', dayfirst=True)

    data = {'X': df_secd, 'df_funda': df_funda, 'y': y}
    return data


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    k = 1
    while not isPD(A3):
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def serialize_solution(name_suffix, solution, runtime):
    result = {
        'solution' : solution.x,
        'objective' : solution.obj,
        'primal_residual' :solution.primal_residual(),
        'dual_residual' : solution.dual_residual(),
        'duality_gap' : solution.duality_gap(),
        'runtime' : runtime
    }

    with open(f'{name_suffix}.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

def to_numpy(data):
    return data.to_numpy() if hasattr(data, 'to_numpy') else data

def concat_constant_columns(X, extra_cols, value = 0):
    if X is None:
        return None

    if X.ndim == 1:
        to_concat = np.zeros(extra_cols) if value == 0 else np.full(extra_cols, value)
        X_new = np.append(X, to_concat)
    else:
        new_shape = (X.shape[0], X.shape[1] + extra_cols)
        X_new = np.zeros(new_shape) if value == 0 else np.full(new_shape, value)
        X_new[0:X.shape[0], 0:X.shape[1]] = X

    return X_new