import numpy as np
import pandas as pd
import qpsolvers
from qpsolvers import solve_qp
import scipy
from helper_functions import nearestPD, to_numpy
from covariance import Covariance
from constraints import Constraints
from optimization_data import OptimizationData



# This class converts an financial optimization problem to a standard quadratic optimization.
# This is the last step before passing problems to solvers.


class QuadraticProgram(dict):

    def __init__(self, *args, **kwargs):
        super(QuadraticProgram, self).__init__(*args, **kwargs)
        self.solver = self['params']['solver_name']

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
        if self.solver in ['ecos', 'scs', 'clarabel']:
            if self.get('b').size == 1 :
                self['b'] = np.array(self.get('b')).reshape(-1)
        problem = qpsolvers.Problem(P = self.get('P'),
                                    q = self.get('q'),
                                    G = self.get('G'),
                                    h = self.get('h'),
                                    A = self.get('A'),
                                    b = self.get('b'),
                                    lb = self.get('lb'),
                                    ub = self.get('ub'))
        # Convert to sparse matrices for best performance
        if self.solver in ['clarabel', 'ecos','gurobi', 'mosek', 'highs', 'qpalm', 'osqp', 'qpswift', 'scs']:
            if self['params']['sparse']:
                if problem.P is not None:
                    problem.P = scipy.sparse.csc_matrix(problem.P)
                if problem.A is not None:
                    problem.A = scipy.sparse.csc_matrix(problem.A)
                if problem.G is not None:
                    problem.G = scipy.sparse.csc_matrix(problem.G)
        solution = qpsolvers.solve_problem(problem = problem,
                                           solver = self.solver,
                                           initvals = self.get('x0'),
                                           verbose = False)
        self['solution'] = solution
        return None

    # 0.5 * x' * P * x + q' * x + const
    def objective_value(self, x: np.ndarray) -> float:
        const = 0 if self.get('constant') is None else to_numpy(self['constant']).item()
        return (0.5 * (x @ self.get('P') @ x) + self.get('q') @ x).item() + const
