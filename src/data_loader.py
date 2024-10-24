'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard

Licensed under GNU LGPL.3, see LICENCE file
'''



import os
from typing import Optional, Union, Any
import pandas as pd
import pickle


def load_pickle(filename: str,
                path: Optional[str] = None) -> Union[Any, None]:
    if path is not None:
        filename = os.path.join(path, filename)
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except EOFError:
        print("Error: Ran out of input. The file may be empty or corrupted.")
        return None
    except Exception as ex:
        print("Error during unpickling object:", ex)
    return None


def load_data_msci(path: str = None, n: int = 24) -> dict[str, pd.DataFrame]:

    '''Loads MSCI daily returns data from 1999-01-01 to 2023-04-18'''

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

    return {'return_series': X, 'bm_series': y}

