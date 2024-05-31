
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







# https://github.com/qpsolvers/qpsolvers







class OptimizationParameter(dict):

    def __init__(self, *args, **kwargs):
        super(OptimizationParameter, self).__init__(*args, **kwargs)
        self.__dict__ = self
        if self.get('solver_name') is None: self['solver_name'] = 'cvxopt'
        if self.get('verbose') is None: self['verbose'] = True
        if self.get('allow_suboptimal') is None: self['allow_suboptimal'] = False


class Objective(dict):

    def __init__(self, *args, **kwargs):
        super(Objective, self).__init__(*args, **kwargs)


class QuadraticProgram(dict):

    def __init__(self, *args, **kwargs):
        super(QuadraticProgram, self).__init__(*args, **kwargs)
        if self.get('sparse') is None:
            self['sparse'] = True

    def linearize_turnover_constraint(self,
                                      x_init: np.ndarray,
                                      to_budget = float('inf')) -> None:
        # Dimensions
        n = len(self.get('q'))
        m = 0 if self.get('G') is None else self.get('G').shape[0]

        # Objective
        if self.get('P') is not None:
            P = np.zeros(shape = (2*n, 2*n))
            P[0:n, 0:n] = self.get('P')
            P = nearestPD(P)
        q = np.append(self.get('q'), np.zeros(n))

        # Inequality constraints
        G = np.zeros(shape = (m+2*n+1, 2*n))
        if self.get('G') is not None:
            G[0:m, 0:n] = self.get('G')
        G[m:(m+n), 0:n] = np.eye(n)
        G[m:(m+n), n:(2*n)] = np.eye(n) * (-1)
        G[(m+n):(m+2*n), 0:n] = np.eye(n) * (-1)
        G[(m+n):(m+2*n), n:(2*n)] = np.eye(n) * (-1)
        G[(m+2*n), ] = np.append(np.zeros(n), np.ones(n))
        h = self.get('h') if self.get('h') is not None else np.empty(shape=(0,))
        h = np.append(h, np.append(np.append(x_init, x_init * (-1)), to_budget))

        # Equality constraints
        if self.get('A').ndim == 1:
            A = np.append(self.get('A'), np.zeros(n))
        else:
            A = np.zeros(shape = (self.get('A').shape[0], 2*n))
            A[0:self.get('A').shape[0], 0:n] = self.get('A')

        # Override the original matrices
        self['P'] = P
        self['q'] = q
        self['G'] = G
        self['h'] = h
        self['A'] = A
        if self.get('lb') is not None:
            self['lb'] = np.append(self.get('lb'), np.zeros(n))
        if self.get('ub') is not None:
            self['ub'] = np.append(self.get('ub'), np.full(n, float('inf')))

        return None

    def linearize_leverage_constraint(self,
                                      N = None,
                                      leverage_budget = 2) -> None:
        # Dimensions
        n = len(self.get('q'))
        mG = 0 if self.get('G') is None else self.get('G').shape[0]
        mA = 1 if self.get('A').ndim == 1 else self.get('A').shape[0]

        # Objective
        if self.get('P') is not None:
            P = np.zeros(shape = (n+2*N, n+2*N))
            P[0:n, 0:n] = self.get('P')
            P = nearestPD(P)
        q = np.append(self.get('q'), np.zeros(2*N))

        # Inequality constraints
        G = np.zeros(shape = (mG+1, n+2*N))
        if self.get('G') is not None:
            G[0:mG, 0:n] = self.get('G')
        G[mG, ] = np.append(np.zeros(n), np.ones(2*N))
        h = self.get('h') if self.get('h') is not None else np.empty(shape=(0,))
        h = np.append(h, leverage_budget)

        # Equality constraints
        A = np.zeros(shape = (mA+N, n+2*N))
        A[0:mA, 0:n] = self.get('A')
        A[mA:(mA+N), 0:N] = np.eye(N)
        A[mA:(mA+N), n:(n+N)] = np.eye(N)
        A[mA:(mA+N), (n+N):(n+2*N)] = np.eye(N) * (-1)
        b = np.append(self.get('b'), np.zeros(N))

        # Override the original matrices
        self['P'] = P
        self['q'] = q
        self['G'] = G
        self['h'] = h
        self['A'] = A
        self['b'] = b
        if self.get('lb') is not None:
            self['lb'] = np.append(self.get('lb'), np.zeros(2*N))
        if self.get('ub') is not None:
            self['ub'] = np.append(self.get('ub'), np.full(2*N, float('inf')))

        return None

    def linearize_turnover_objective(self,
                                     x_init: np.ndarray,
                                     transaction_cost = 0.002) -> None:
        # Dimensions
        n = len(self.get('q'))
        m = 0 if self.get('G') is None else self.get('G').shape[0]

        # Objective
        if self.get('P') is not None:
            P = np.zeros(shape = (2*n, 2*n))
            P[0:n, 0:n] = self.get('P')
            P = nearestPD(P)
        q = np.append(self.get('q'), np.full(n, transaction_cost))

        # Inequality constraints
        G = np.zeros(shape = (m+2*n, 2*n))
        if self.get('G') is not None:
            G[0:m, 0:n] = self.get('G')
        G[m:(m+n), 0:n] = np.eye(n)
        G[m:(m+n), n:(2*n)] = np.eye(n) * (-1)
        G[(m+n):(m+2*n), 0:n] = np.eye(n) * (-1)
        G[(m+n):(m+2*n), n:(2*n)] = np.eye(n) * (-1)
        h = self.get('h') if self.get('h') is not None else np.empty(shape=(0,))
        h = np.append(h, np.append(x_init, x_init * (-1)))

        # Equality constraints
        if self.get('A').ndim == 1:
            A = np.append(self.get('A'), np.zeros(n))
        else:
            A = np.zeros(shape = (self.get('A').shape[0], 2*n))
            A[0:self.get('A').shape[0], 0:n] = self.get('A')

        # Override the original matrices
        self['P'] = P
        self['q'] = q
        self['G'] = G
        self['h'] = h
        self['A'] = A
        if self.get('lb') is not None:
            self['lb'] = np.append(self.get('lb'), np.zeros(n))
        if self.get('ub') is not None:
            self['ub'] = np.append(self.get('ub'), np.full(n, 10**6))

        return None

    def solve(self) -> None:

        problem = qpsolvers.Problem(P = self.get('P'),
                                    q = self.get('q'),
                                    G = self.get('G'),
                                    h = self.get('h'),
                                    A = self.get('A'),
                                    b = self.get('b'),
                                    lb = self.get('lb'),
                                    ub = self.get('ub'))
        # Convert to sparse matrices for best performance
        if self['solver_name'] in ['highs', 'qpalm', 'osqp']:
            if self['sparse']:
                if problem.P is not None:
                    problem.P = scipy.sparse.csc_matrix(problem.P)
                if problem.A is not None:
                    problem.A = scipy.sparse.csc_matrix(problem.A)
                if problem.G is not None:
                    problem.G = scipy.sparse.csc_matrix(problem.G)
        solution = qpsolvers.solve_problem(problem = problem,
                                           solver = self.get('solver_name'),
                                           initvals = self.get('x0'),
                                           verbose = False)
        self['solution'] = solution
        return None

    def objective_value(self, x: np.ndarray) -> float:
        constant = 0 if self.get('constant') is None else self.get('constant')
        return 0.5 * (x @ self.get('P') @ x) + self.get('q') @ x + constant


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
        weights = pd.Series(self.model['solution'].x[0:len(self.constraints.selection)],
                            index = self.constraints.selection)
        self.results = {'weights': weights.to_dict(),
                        'status': self.model['solution'].found}
        return None

    def model_qpsolvers(self) -> None:
        GhAb = self.constraints.to_GhAb()
        self.model = QuadraticProgram(P = self.objective['P'],
                                      q = self.objective['q'],
                                      constant = self.objective.get('constant'),
                                      G = GhAb['G'],
                                      h = GhAb['h'],
                                      A = GhAb['A'],
                                      b = GhAb['b'],
                                      lb = self.constraints.box['lower'].to_numpy(),
                                      ub = self.constraints.box['upper'].to_numpy(),
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
        y = np.log(1 + optimization_data['y'])

        P = 2 * (X.T @ X)
        q = -2 * X.T @ y
        constant = y.T @ y

        l2_penalty = self.params.get('l2_penalty')
        if l2_penalty is not None:
            P = 2 * ((X.T @ X) + l2_penalty * np.eye(X.shape[1]))

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




