# GeoFin : a python library for portfolio optimization and index replication
# GeoFin is part of GeomScale project

# Copyright (c) 2024 Cyril Bachelard
# Copyright (c) 2024 Minh Ha Ho

# Licensed under GNU LGPL.3, see LICENCE file



import os
from typing import Dict
import pandas as pd


def load_data(universe, path: str = None):
    if universe == 'msci':
        data = load_data_msci()
    elif universe == 'usa':
        data = load_data_usa()
    else:
        raise ValueError('Universe not recognized.')
    return data

# MSCI daily returns data from 1999-01-01 to 2023-04-18
def load_data_msci(path: str = None, n: int = 24) -> Dict[str, pd.DataFrame]:

    path = os.path.join(os.getcwd(), f'data{os.sep}') if path is None else path
    # Load msci country index return series
    df = pd.read_csv(os.path.join(path, 'msci_country_indices.csv'),
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

    return {'X': X, 'y': y}

# USA daily returns
# SPTR.csv from 1996-01-20 to 2023-06-06
# usa_returns.csv from 2005-01-03 to 2024-01-22
def load_data_usa(path: str = None) -> Dict[str, pd.DataFrame]:

    # Load U.S. security data
    path = os.path.join(os.getcwd(), f'data{os.sep}') if not path else path
    df_secd = pd.read_csv(os.path.join(path, 'usa_returns.csv'), index_col = 0, parse_dates=True)
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

    return {'X': df_secd, 'df_funda': df_funda, 'y': y}
