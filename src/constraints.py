
############################################################################
### CONSTRAINTS
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard, Minhha Ho
# This version:     24.05.2024
# First version:    24.05.2024
# --------------------------------------------------------------------------



from typing import Dict
import pandas as pd
import numpy as np







# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------


def match_arg(x, lst):
    return [el for el in lst if x in el][0]


def box_constraint(box_type = "LongOnly",
                   lower = None,
                   upper = None) -> dict:

    box_type = match_arg(box_type, ["LongOnly", "LongShort", "Unbounded"])

    if box_type == "Unbounded":
        if lower is None:
            lower = float("-inf")
        if upper is None:
            upper = float("inf")
    elif box_type == "LongShort":
        if lower is None:
            lower = -1
        if upper is None:
            upper = 1
    else:
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
            if upper is None:
                upper = lower * 0 + 1

    ans = {'box_type': box_type,
           'lower': lower,
           'upper': upper}
    return ans


def linear_constraint(Amat = None,
                      sense = "=",
                      rhs = float("inf"),
                      index_or_name = None,
                      a_values = None) -> dict:
    ans = {'Amat': Amat,
           'sense': sense,
           'rhs': rhs}
    if index_or_name is not None:
        ans['index_or_name'] = index_or_name
    if a_values is not None:
        ans['a_values'] = a_values
    return ans




# --------------------------------------------------------------------------
# Class definition
# --------------------------------------------------------------------------

class Constraints:

    def __init__(self, selection = "NA") -> None:

        if not all(isinstance(item, str) for item in selection):
            raise ValueError("argument 'selection' has to be a character vector.")
        self.selection = selection
        self.budget = {'Amat': pd.DataFrame(dtype='float64'), 'sense': None, 'rhs': None}
        self.box = {'type': 'NA', 'lower': pd.Series(dtype='float64'), 'upper': pd.Series(dtype='float64')}
        self.linear = {'Amat': pd.DataFrame(dtype='float64'), 'sense': pd.Series(dtype='float64'), 'rhs': pd.Series(dtype='float64')}
        self.l1 = {}
        return None

    def __str__(self) -> str:
        txt = ''
        for key in vars(self).keys():
            txt = txt + f'\n{key}:\n\n{vars(self)[key]}\n'
        return txt

    def add_budget(self, rhs = 1, sense = '=') -> None:
        if self.budget.get('rhs') is not None:
            print("Existing budget constraint is overwritten\n")
        a_values = pd.Series([1] * len(self.selection), index = self.selection)
        self.budget = {'Amat': a_values,
                       'sense': sense,
                       'rhs': rhs}
        return None

    def add_box(self,
                box_type = "LongOnly",
                lower = None,
                upper = None) -> None:

        box_type = match_arg(box_type, ["LongOnly", "LongShort", "Unbounded"])
        boxcon = box_constraint(box_type = box_type,
                                lower = lower,
                                upper = upper)

        if np.isscalar(boxcon['lower']):
            boxcon['lower'] = pd.Series(np.repeat(boxcon['lower'], len(self.selection)), index=self.selection)
        if np.isscalar(boxcon['upper']):
            boxcon['upper'] = pd.Series(np.repeat(boxcon['upper'], len(self.selection)), index=self.selection)

        if not (self.box['upper'] > self.box['lower']).all():
            raise ValueError("Some lower bounds are higher than the corresponding upper bounds.")

        self.box = boxcon
        return None

    def add_linear(self,
                   Amat: pd.DataFrame(dtype='float64') = None,
                   a_values: pd.Series(dtype='float64') = None,
                   sense: pd.Series(dtype='float64') = None,
                   rhs: pd.Series(dtype='float64') = None,
                   name: str = None) -> None:
        if Amat is None:
            if a_values is None:
                raise ValueError("Either 'Amat' or 'a_values' must be provided.")
            else:
                Amat = pd.DataFrame(a_values).T.reindex(columns = self.selection).fillna(0)
                if name is not None:
                    Amat.index = [name]
        if not self.linear['Amat'].empty:
            Amat = pd.concat([self.linear['Amat'], Amat], axis = 0, ignore_index = False)
            sense = pd.concat([self.linear['sense'], sense], axis = 0, ignore_index = False)
            rhs = pd.concat([self.linear['rhs'], rhs], axis = 0, ignore_index = False)
        self.linear = {'Amat': Amat,
                      'sense': sense,
                      'rhs': rhs}
        return None

    def add_l1(self,
               name: str,
               x0 = None,
               rhs = None,
               *args, **kwargs) -> None:
        if rhs is None:
            raise TypeError("argument 'rhs' is required.")
        con = {'rhs': rhs}
        if not x0 is None:
            con['x0'] = x0
        for i, arg in enumerate(args):
            con[f'arg{i}'] = arg
        for key, value in kwargs.items():
            con[key] = value
        self.l1[name] = con
        return None

    def to_GhAb(self, lbub_to_G = False) -> Dict[str, pd.DataFrame]:
        A = None
        b = None
        G = None
        h = None
        if not self.budget['Amat'].empty:
            if self.budget['sense'] == '=':
                A = np.array(self.budget['Amat'], dtype = float)
                b = np.array(self.budget['rhs'], dtype = float)
            else:
                G = np.array(self.budget['Amat'], dtype = float)
                h = np.array(self.budget['rhs'], dtype = float)

        if lbub_to_G:
            I = np.diag(np.ones(len(self.selection)))
            G_tmp = np.concatenate((-I, I), axis = 0)
            h_tmp = np.concatenate((-self.box["lower"], self.box["upper"]), axis = 0)
            G = np.vstack((G, G_tmp)) if (G is not None) else G_tmp
            h = np.concatenate((h, h_tmp), axis = None) if (h is not None) else h_tmp

        if not self.linear['Amat'].empty:
            # Extract equality constraints
            idx_eq = np.array(self.linear['sense'] == '=')
            if idx_eq.sum() > 0:
                A_tmp = self.linear['Amat'][idx_eq].to_numpy()
                b_tmp = self.linear['rhs'][idx_eq].to_numpy()
                A = np.vstack((A, A_tmp)) if (A is not None) else A_tmp
                b = np.concatenate((b, b_tmp), axis = None) if (b is not None) else b_tmp
                if idx_eq.sum() < self.linear['Amat'].shape[0]:
                    G_tmp = self.linear['Amat'][idx_eq == False].to_numpy()
                    h_tmp = self.linear['rhs'][idx_eq == False].to_numpy()
            else:
                G_tmp = self.linear['Amat'].to_numpy()
                h_tmp = self.linear['rhs'].to_numpy()

            # Ensure that the system of inequalities is all '<='
            idx_geq = np.array(self.linear['sense'] == '>=')
            if idx_geq.sum() > 0:
                G_tmp[idx_geq] = G_tmp[idx_geq] * (-1)
                h_tmp[idx_geq] = h_tmp[idx_geq] * (-1)

            if 'G_tmp' in locals():
                G = np.vstack((G, G_tmp)) if (G is not None) else G_tmp
                h = np.concatenate((h, h_tmp), axis = None) if (h is not None) else h_tmp

        ans = {'G': G, 'h': h, 'A': A, 'b': b}
        return ans