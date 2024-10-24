'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''


############################################################################
### CLASS Selection
############################################################################



from typing import Union, Optional
import pandas as pd



class Selection:

    def __init__(self, ids: pd.Index = pd.Index([])):
        self._filtered: dict[str, Union[pd.Series, pd.DataFrame]] = {}
        self.selected = ids

    @property
    def selected(self) -> pd.Index:
        return self._selected

    @selected.setter
    def selected(self, value):
        if not isinstance(value, pd.Index):
            raise ValueError(
                "Inconsistent input type for selected.setter. Needs to be a pd.Index."
            )
        self._selected = value

    @property
    def filtered(self):
        return self._filtered

    def get_selected(self, filter_names: Optional[list[str]] = None) -> pd.Index:
        if filter_names is not None:
            df = self.df_binary(filter_names)
        else:
            df = self.df_binary()
        return df[df.eq(1).all(axis=1)].index

    def clear(self) -> None:
        self.selected = pd.Index([])
        self._filtered = {}

    def add_filtered(self,
                     filter_name: str,
                     value: Union[pd.Series, pd.DataFrame]) -> None:

        # Check input types
        if not isinstance(filter_name, str) or not filter_name.strip():
            raise ValueError("Argument 'filter_name' must be a nonempty string.")

        if not isinstance(value, pd.Series) and not isinstance(value, pd.DataFrame):
            raise ValueError(
                'Inconsistent input type. Needs to be a pd.Series or a pd.DataFrame.'
            )

        # Ensure that column 'binary' is of type int if it exists
        if isinstance(value, pd.Series):
            if value.name == 'binary':
                if not value.isin([0, 1]).all():
                    raise ValueError("Column 'binary' must contain only 0s and 1s.")
                else:
                    value = value.astype(int)

        if isinstance(value, pd.DataFrame):
            if 'binary' in value.columns:
                if not value['binary'].isin([0, 1]).all():
                    raise ValueError("Column 'binary' must contain only 0s and 1s.")
                else:
                    value['binary'] = value['binary'].astype(int)

        # Add to filtered
        self._filtered[filter_name] = value

        # Reset selected
        self.selected = self.get_selected()
        return None

    def df(self, filter_names: Optional[list[str]] = None) -> pd.DataFrame:

        if filter_names is None:
            filter_names = self.filtered.keys()
        return pd.concat(
            {
                key: (
                    pd.DataFrame(self.filtered[key])
                    if isinstance(self.filtered[key], pd.Series)
                    else self.filtered[key]
                )
                for key in filter_names
            },
            axis = 1,
        )

    def df_binary(self, filter_names: Optional[list[str]] = None) -> pd.DataFrame:

        if filter_names is None:
            filter_names = self.filtered.keys()
        df = self.df(filter_names = filter_names).filter(like = 'binary').dropna()
        df.columns = df.columns.droplevel(1)
        return df
