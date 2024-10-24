'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''




import warnings
import pandas as pd
import numpy as np
from typing import Dict





class Constraints:

    def __init__(self, selection="NA") -> None:
        if not all(isinstance(item, str) for item in selection):
            raise ValueError("argument 'selection' has to be a character vector.")

        self.selection = selection
        self.budget = {'Amat': None, 'sense': None, 'rhs': None}
        self.box = {'box_type': 'NA', 'lower': None, 'upper': None}
        self.linear = {'Amat': None, 'sense': None, 'rhs': None}
        self.l1 = {}
        return None

    def __str__(self) -> str:
        return ' '.join(f'\n{key}:\n\n{vars(self)[key]}\n' for key in vars(self).keys())

    def add_budget(self, rhs=1, sense='=') -> None:
        if self.budget.get('rhs') is not None:
            warnings.warn("Existing budget constraint is overwritten\n")

        a_values = pd.Series(np.ones(len(self.selection)), index=self.selection)
        self.budget = {'Amat': a_values,
                       'sense': sense,
                       'rhs': rhs}
        return None

    def add_box(self,
                box_type="LongOnly",
                lower=None,
                upper=None) -> None:
        boxcon = box_constraint(box_type, lower, upper)

        if np.isscalar(boxcon['lower']):
            boxcon['lower'] = pd.Series(np.repeat(float(boxcon['lower']), len(self.selection)), index=self.selection)
        if np.isscalar(boxcon['upper']):
            boxcon['upper'] = pd.Series(np.repeat(float(boxcon['upper']), len(self.selection)), index=self.selection)

        if (boxcon['upper'] < boxcon['lower']).any():
            raise ValueError("Some lower bounds are higher than the corresponding upper bounds.")

        self.box = boxcon
        return None

    def add_linear(self,
                   Amat: pd.DataFrame = None,
                   a_values: pd.Series = None,
                   sense: str = '=',
                   rhs=None,
                   name: str = None) -> None:
        if Amat is None:
            if a_values is None:
                raise ValueError("Either 'Amat' or 'a_values' must be provided.")
            else:
                Amat = pd.DataFrame(a_values).T.reindex(columns=self.selection).fillna(0)
                if name is not None:
                    Amat.index = [name]

        if isinstance(sense, str):
            sense = pd.Series([sense])

        if isinstance(rhs, (int, float)):
            rhs = pd.Series([rhs])

        if self.linear['Amat'] is not None:
            Amat = pd.concat([self.linear['Amat'], Amat], axis=0, ignore_index=False)
            sense = pd.concat([self.linear['sense'], sense], axis=0, ignore_index=False)
            rhs = pd.concat([self.linear['rhs'], rhs], axis=0, ignore_index=False)

        Amat.fillna(0, inplace=True)

        self.linear = {'Amat': Amat, 'sense': sense, 'rhs': rhs}
        return None

    # name: turnover or leverage
    def add_l1(self,
               name: str,
               rhs=None,
               x0=None,
               *args, **kwargs) -> None:
        if rhs is None:
            raise TypeError("argument 'rhs' is required.")
        con = {'rhs': rhs}
        if x0:
            con['x0'] = x0
        for i, arg in enumerate(args):
            con[f'arg{i}'] = arg
        for key, value in kwargs.items():
            con[key] = value
        self.l1[name] = con
        return None

    def to_GhAb(self, lbub_to_G: bool = False) -> Dict[str, pd.DataFrame]:
        A = None
        b = None
        G = None
        h = None

        if self.budget['Amat'] is not None:
            if self.budget['sense'] == '=':
                A = np.array(self.budget['Amat'], dtype=float)
                b = np.array(self.budget['rhs'], dtype=float)
            else:
                G = np.array(self.budget['Amat'], dtype=float)
                h = np.array(self.budget['rhs'], dtype=float)

        if lbub_to_G:
            I = np.eye(len(self.selection))
            G_tmp = np.concatenate((-I, I), axis=0)
            h_tmp = np.concatenate((-self.box["lower"], self.box["upper"]), axis=0)
            G = np.vstack((G, G_tmp)) if (G is not None) else G_tmp
            h = np.concatenate((h, h_tmp), axis=None) if h is not None else h_tmp

        if self.linear['Amat'] is not None:
            Amat = self.linear['Amat'].copy()
            rhs = self.linear['rhs'].copy()

            # Ensure that the system of inequalities is all '<='
            idx_geq = np.array(self.linear['sense'] == '>=')
            if idx_geq.sum() > 0:
                Amat[idx_geq] = -Amat[idx_geq]
                rhs[idx_geq] = -rhs[idx_geq]

            # Extract equality constraints
            idx_eq = np.array(self.linear['sense'] == '=')
            if idx_eq.sum() > 0:
                A_tmp = Amat[idx_eq].to_numpy()
                b_tmp = rhs[idx_eq].to_numpy()
                A = np.vstack((A, A_tmp)) if A is not None else A_tmp
                b = np.concatenate((b, b_tmp), axis=None) if b is not None else b_tmp
                if idx_eq.sum() < Amat.shape[0]:
                    G_tmp = Amat[idx_eq == False].to_numpy()
                    h_tmp = rhs[idx_eq == False].to_numpy()
            else:
                G_tmp = Amat.to_numpy()
                h_tmp = rhs.to_numpy()

            if 'G_tmp' in locals():
                G = np.vstack((G, G_tmp)) if G is not None else G_tmp
                h = np.concatenate((h, h_tmp), axis=None) if h is not None else h_tmp

        # To ensure A and G are matrices (even with only 1 row)
        A = A.reshape(-1, A.shape[-1]) if A is not None else None
        G = G.reshape(-1, G.shape[-1]) if G is not None else None

        return {'G': G, 'h': h, 'A': A, 'b': b}



# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def match_arg(x, lst):
    return [el for el in lst if x in el][0]

def box_constraint(box_type="LongOnly",
                   lower=None,
                   upper=None) -> dict:
    box_type = match_arg(box_type, ["LongOnly", "LongShort", "Unbounded"])

    if box_type == "Unbounded":
        lower = float("-inf") if lower is None else lower
        upper = float("inf") if upper is None else upper
    elif box_type == "LongShort":
        lower = -1 if lower is None else lower
        upper = 1 if upper is None else upper
    elif box_type == "LongOnly":
        if lower is None:
            if upper is None:
                lower = 0
                upper = 1
            else:
                lower = upper * 0
        else:
            if not np.isscalar(lower):
                if any(l < 0 for l in lower):
                    raise ValueError("Inconsistent lower bounds for box_type 'LongOnly'. "
                                     "Change box_type to LongShort or ensure that lower >= 0.")

            upper = lower * 0 + 1 if upper is None else upper

    return {'box_type': box_type, 'lower': lower, 'upper': upper}

def linear_constraint(Amat=None,
                      sense="=",
                      rhs=float("inf"),
                      index_or_name=None,
                      a_values=None) -> dict:
    ans = {'Amat': Amat,
           'sense': sense,
           'rhs': rhs}
    if index_or_name is not None:
        ans['index_or_name'] = index_or_name
    if a_values is not None:
        ans['a_values'] = a_values
    return ans

