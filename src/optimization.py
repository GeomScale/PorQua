# GeoFin : a python library for portfolio optimization and index replication
# GeoFin is part of GeomScale project

# Copyright (c) 2024 Cyril Bachelard
# Copyright (c) 2024 Minh Ha Ho

# Licensed under GNU LGPL.3, see LICENCE file



from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from qpsolvers import solve_qp
import scipy

from helper_functions import to_numpy
from covariance import Covariance
from constraints import Constraints
from optimization_data import OptimizationData
import qp_problems

# https://github.com/qpsolvers/qpsolvers


class OptimizationParameter(dict):

    def __init__(self, *args, **kwargs):
        super(OptimizationParameter, self).__init__(*args, **kwargs)
        self.__dict__ = self
        if not self.get('solver_name'): self['solver_name'] = 'cvxopt'
        if not self.get('verbose'): self['verbose'] = True
        if not self.get('allow_suboptimal'): self['allow_suboptimal'] = False


class Objective(dict):

    def __init__(self, *args, **kwargs):
        super(Objective, self).__init__(*args, **kwargs)

class Optimization(ABC):

    def __init__(self,
                 params: OptimizationParameter = None,
                 constraints: Constraints = None,
                 *args, **kwargs):
        self.params = OptimizationParameter(*args, **kwargs) if params is None else params
        self.objective = Objective()
        self.constraints = Constraints() if constraints is None else constraints
        self.model = None
        self.results = None

    @abstractmethod
    def solve(self) -> bool:

        # Ensure that P and q are numpy arrays
        if 'P' in self.objective.keys():
            P = to_numpy(self.objective['P'])
        else:
            raise ValueError("Missing matrix 'P' in objective.")
        if 'q' in self.objective.keys():
            q = to_numpy(self.objective['q'])
        else:
            q = np.zeros(len(self.constraints.selection))

        self.objective['P'] = P
        self.objective['q'] = q

        self.solve_qpsolvers()
        return self.results['status']

    def solve_qpsolvers(self) -> None:
        self.model_qpsolvers()
        self.model.solve()
        universe = self.constraints.selection
        solution = self.model['solution']
        status = solution.found
        weights = pd.Series(solution.x[:len(universe)] if status else [None] * len(universe),
                                index = universe)

        self.results = {'weights': weights.to_dict(),
                        'status': self.model['solution'].found}

        return None

    def model_qpsolvers(self) -> None:
        universe = self.constraints.selection

        GhAb = self.constraints.to_GhAb()

        lb = self.constraints.box['lower'].to_numpy() if not self.constraints.box['box_type'] == 'NA' else None
        ub = self.constraints.box['upper'].to_numpy() if not self.constraints.box['box_type'] == 'NA' else None

        self.model = qp_problems.QuadraticProgram(P = self.objective['P'],
                                        q = self.objective['q'],
                                        constant = self.objective.get('constant'),
                                        G = GhAb['G'],
                                        h = GhAb['h'],
                                        A = GhAb['A'],
                                        b = GhAb['b'],
                                        lb = lb,
                                        ub = ub,
                                        params = self.params)

        # Choose which reference position to be used
        tocon = self.constraints.l1.get('turnover')
        x0 = tocon['x0'] if tocon is not None and tocon.get('x0') is not None else self.params.get('x0')
        x_init = {asset: x0.get(asset, 0) for asset in universe} if x0 is not None else None

        # Transaction cost in the objective
        transaction_cost = self.params.get('transaction_cost')
        if transaction_cost is not None and x_init is not None:
            self.model.linearize_turnover_objective(pd.Series(x_init), transaction_cost)

        # Turnover constraint
        if tocon and not transaction_cost and x_init is not None:
            self.model.linearize_turnover_constraint(pd.Series(x_init), tocon['rhs'])

        # Leverage constraint
        levcon = self.constraints.l1.get('leverage')
        if levcon is not None:
            self.model.linearize_leverage_constraint(N = len(universe), leverage_budget = levcon['rhs'])
        return None



class LeastSquares(Optimization):

    def __init__(self,
                 covariance: Optional[Covariance] = None,
                 *arg, **kwarg):
        self.covariance = covariance
        super().__init__(*arg, **kwarg)
        if self.params.get('l2_penalty') is None:
            self.params['l2_penalty'] = 0

    def set_objective(self, optimization_data: OptimizationData) -> None:

        X = np.log(1 + optimization_data['X'])
        y = np.log(1 + optimization_data['y'])

        # 0.5 * w * P * w' - q * w' + constant
        P = 2 * (X.T @ X)
        q = to_numpy(-2 * X.T @ y).reshape((-1,))
        constant = to_numpy(y.T @ y).item()

        l2_penalty = self.params.get('l2_penalty')
        if l2_penalty is not None and l2_penalty != 0:
            P = P + 2 * l2_penalty * np.eye(X.shape[1])

        self.objective = Objective(P = P, q = q, constant = constant)
        return None

    def solve(self) -> bool:
        return super().solve()



class WeightedLeastSquares(Optimization):

    def set_objective(self, optimization_data: OptimizationData) -> None:

        X = np.log(1 + optimization_data['X'])
        y = np.log(1 + optimization_data['y'])

        tau = self.params['tau']
        lambda_val = np.exp(-np.log(2) / tau)
        i = np.arange(X.shape[0])
        wt_tmp = lambda_val ** i
        wt = np.flip(wt_tmp / np.sum(wt_tmp) * len(wt_tmp))
        W = np.diag(wt)

        P = 2 * ((X.T).to_numpy() @ W @ X)
        q = -2 * (X.T).to_numpy() @ W @ y
        constant = (y.T).to_numpy() @ W @ y

        self.objective = Objective(P = P, q = q, constant = constant)
        return None

    def solve(self) -> bool:
        return super().solve()



class QEQW(Optimization):

    def __init__(self, *arg, **kwarg):
        covariance = Covariance(method = 'duv')
        super().__init__(covariance = covariance, *arg, **kwarg)

    def set_objective(self, optimization_data: OptimizationData) -> None:
        covmat = self.covariance.estimate(X = optimization_data['X']) * 2
        mu = np.zeros(optimization_data['X'].shape[1])
        self.objective = Objective(P = covmat, q = mu)
        return None

    def solve(self) -> bool:
        return super().solve()
