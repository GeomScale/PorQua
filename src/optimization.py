
############################################################################
### OPTIMIZATION
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard, Minhha Ho
# This version:     24.05.2024
# First version:    24.05.2024
# --------------------------------------------------------------------------



from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import qpsolvers
from qpsolvers import solve_qp
import scipy



from helper_functions import nearestPD
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
                 constraints = None,
                 *args, **kwargs):
        self.params = OptimizationParameter(*args, **kwargs) if params is None else params
        self.objective = Objective()
        self.constraints = Constraints() if constraints is None else constraints
        self.model = None
        self.results = None

    @abstractmethod
    def solve(self) -> None:

        # Ensure that P and q are numpy arrays
        if 'P' in self.objective.keys():
            P = self.objective['P']
            if hasattr(P, "to_numpy"):
                P = P.to_numpy()
        else:
            # P = 1 + np.zeros(shape = (len(self.constraints.selection), len(self.constraints.selection)))
            raise ValueError("Missing matrix 'P' in objective.")
        if 'q' in self.objective.keys():
            q = self.objective['q']
            if hasattr(q, "to_numpy"):
                q = q.to_numpy()
        else:
            q = np.zeros(len(self.constraints.selection))
        self.objective['P'] = P
        self.objective['q'] = q

        self.solve_qpsolvers()
        return None

    def solve_qpsolvers(self) -> None:
        self.model_qpsolvers()
        self.model.solve()
        status = self.model['solution'].found

        weights = pd.Series(self.model['solution'].x[0:len(self.constraints.selection)],
                                index = self.constraints.selection) if status else \
                                    pd.Series([None] * len(self.constraints.selection),
                                        index = self.constraints.selection)

        self.results = {'weights': weights.to_dict(),
                        'status': self.model['solution'].found}

        return None

    def model_qpsolvers(self) -> None:
        GhAb = self.constraints.to_GhAb()

        if self.constraints.box['box_type'] == 'NA':
            lb = None
            ub = None
        else:
            lb = self.constraints.box['lower'].to_numpy()
            ub = self.constraints.box['upper'].to_numpy()

        self.model = qp_problems.QuadraticProgram(P = self.objective['P'],
                                        q = self.objective['q'],
                                        constant = self.objective.get('constant'),
                                        G = GhAb['G'],
                                        h = GhAb['h'],
                                        A = GhAb['A'],
                                        b = GhAb['b'],
                                        lb = lb,
                                        ub = ub,
                                        solver_name = self.params['solver_name'])

        # Transaction cost in the objective
        transaction_cost = self.params.get('transaction_cost')
        if transaction_cost is not None:
            tocon = self.constraints.l1['turnover']
            x_init = np.array(list(tocon['x0'].values()))
            self.model.linearize_turnover_objective(x_init = x_init,
                                                    transaction_cost = transaction_cost)

        # Turnover constraint
        tocon = self.constraints.l1.get('turnover')
        if tocon is not None and transaction_cost is None:
            x_init = np.array(list(tocon['x0'].values()))
            self.model.linearize_turnover_constraint(x_init = x_init,
                                                     to_budget = tocon['rhs'])

        # Leverage constraint
        levcon = self.constraints.l1.get('leverage')
        if levcon is not None:
            self.model.linearize_leverage_constraint(N = len(self.constraints.selection),
                                                     leverage_budget = levcon['rhs'])
        return None


class LeastSquares(Optimization):

    def __init__(self,
                 covariance: Optional[Covariance] = None,
                 *arg, **kwarg):
        self.covariance = covariance
        super().__init__(*arg, **kwarg)
        if self.params.get('l2_penalty') == None:
            self.params['l2_penalty'] = 0

    def set_objective(self, optimization_data: OptimizationData) -> None:

        X = np.log(1 + optimization_data['X'])
        y = np.log(1 + optimization_data['y']).flatten()

        # 0.5 * w * P * w' - q * w' + constant
        P = 2 * (X.T @ X)
        q = -2 * X.T @ y
        constant = y.dot(y)

        l2_penalty = self.params.get('l2_penalty')
        if l2_penalty is not None:
            P = P + 2 * l2_penalty * np.eye(X.shape[1])

        self.objective = Objective(y = y,
                                   X = X,
                                   P = P,
                                   q = q,
                                   constant = constant)
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

        self.objective = Objective(y = y,
                                   X = X,
                                   P = P,
                                   q = q,
                                   constant = constant)
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
        self.objective = Objective(q = mu,
                                   P = covmat)
        return None

    def solve(self) -> bool:
        return super().solve()
