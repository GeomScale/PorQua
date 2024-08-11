# GeoFin : a python library for portfolio optimization and index replication
# GeoFin is part of GeomScale project

# Copyright (c) 2024 Cyril Bachelard
# Copyright (c) 2024 Minh Ha Ho

# Licensed under GNU LGPL.3, see LICENCE file


import numpy as np
import qpsolvers
import scipy
import pickle
from helper_functions import isPD, nearestPD

IGNORED_SOLVERS = {'gurobi',  # Restricted license - for non-production use only - expires 2025-11-24
                   'mosek',  # Commercial solver
                   'ecos',  # LinAlgError: 0-dimensional array given. Array must be at least two-dimensional
                   'scs',  # ValueError: Failed to parse cone field bu
                   'piqp',
                   'proxqp',
                   'clarabel'
                   }

SPARSE_SOLVERS = {'clarabel', 'ecos', 'gurobi', 'mosek', 'highs', 'qpalm', 'osqp', 'qpswift', 'scs'}
ALL_SOLVERS = {'clarabel', 'cvxopt', 'daqp', 'ecos', 'gurobi', 'highs', 'mosek', 'osqp', 'piqp', 'proxqp', 'qpalm', 'quadprog', 'scs'}
USABLE_SOLVERS = ALL_SOLVERS - IGNORED_SOLVERS


# This class converts a financial optimization problem to a standard quadratic optimization.
class QuadraticProgram(dict):

    def __init__(self, *args, **kwargs):
        super(QuadraticProgram, self).__init__(*args, **kwargs)
        self.solver = self['params']['solver_name']

    def linearize_turnover_constraint(self, x_init: np.ndarray, to_budget=float('inf')) -> None:
        # Dimensions
        n = len(self.get('q'))
        m = 0 if self.get('G') is None else self.get('G').shape[0]

        # Objective
        P = np.pad(self['P'], (0, n)) if self.get('P') is not None else None
        q = np.pad(self['q'], (0, n)) if self.get('q') is not None else None

        # Inequality constraints
        G = np.zeros(shape=(m + 2 * n + 1, 2 * n))
        if self.get('G') is not None:
            G[0:m, 0:n] = self.get('G')
        G[m:(m + n), 0:n] = np.eye(n)
        G[m:(m + n), n:(2 * n)] = np.eye(n) * (-1)
        G[(m + n):(m + 2 * n), 0:n] = np.eye(n) * (-1)
        G[(m + n):(m + 2 * n), n:(2 * n)] = np.eye(n) * (-1)
        G[(m + 2 * n),] = np.append(np.zeros(n), np.ones(n))
        h = self.get('h') if self.get('h') is not None else np.empty(shape=(0,))
        h = np.append(h, np.append(np.append(x_init, -x_init), to_budget))

        # Equality constraints
        #A = concat_constant_columns(self.get('A'), n)
        A = np.pad(self['A'], [(0, 0), (0, n)]) if self.get('A') is not None else None

        lb = np.pad(self['lb'], (0, n)) if self.get('lb') is not None else None
        ub = np.pad(self['ub'], (0, n), constant_values=float('inf')) if self.get('ub') is not None else None

        # Override the original matrices
        self.update({'P': P,
                     'q': q,
                     'G': G,
                     'h': h,
                     'A': A,
                     'lb': lb,
                     'ub': ub})

        return None

    def linearize_leverage_constraint(self, N=None, leverage_budget=2) -> None:
        # Dimensions
        n = len(self.get('q'))
        mG = 0 if self.get('G') is None else self.get('G').shape[0]
        mA = 1 if self.get('A').ndim == 1 else self.get('A').shape[0]

        # Objective
        P = np.pad(self['P'], (0, 2 * N)) if self.get('P') is not None else None
        q = np.pad(self['q'], (0, 2 * N)) if self.get('q') is not None else None

        # Inequality constraints
        G = np.zeros(shape=(mG + 1, n + 2 * N))
        if self.get('G') is not None:
            G[0:mG, 0:n] = self.get('G')
        G[mG,] = np.append(np.zeros(n), np.ones(2 * N))
        h = self.get('h') if self.get('h') is not None else np.empty(shape=(0,))
        h = np.append(h, leverage_budget)

        # Equality constraints
        A = np.zeros(shape=(mA + N, n + 2 * N))
        A[0:mA, 0:n] = self.get('A')
        A[mA:(mA + N), 0:N] = np.eye(N)
        A[mA:(mA + N), n:(n + N)] = np.eye(N)
        A[mA:(mA + N), (n + N):(n + 2 * N)] = -np.eye(N)
        b = np.pad(self.get('b'), (0, N))

        lb = np.pad(self['lb'], (0, 2 * N)) if self.get('lb') is not None else None
        ub = np.pad(self['ub'], (0, 2 * N), constant_values=float('inf')) if self.get('ub') is not None else None

        # Override the original matrices
        self.update({'P': P,
                     'q': q,
                     'G': G,
                     'h': h,
                     'A': A,
                     'b': b,
                     'lb': lb,
                     'ub': ub})

        return None

    def linearize_turnover_objective(self,
                                     x_init: np.ndarray,
                                     transaction_cost=0.002) -> None:
        # Dimensions
        n = len(self.get('q'))
        m = 0 if self.get('G') is None else self.get('G').shape[0]

        # Objective
        P = np.pad(self['P'], (0, n)) if self.get('P') is not None else None
        q = np.pad(self['q'], (0, n), constant_values=transaction_cost) if self.get('q') is not None else None

        # Inequality constraints
        G = np.zeros(shape=(m + 2 * n, 2 * n))
        if self.get('G') is not None:
            G[0:m, 0:n] = self.get('G')
        G[m:(m + n), 0:n] = np.eye(n)
        G[m:(m + n), n:(2 * n)] = -np.eye(n)
        G[(m + n):(m + 2 * n), 0:n] = -np.eye(n)
        G[(m + n):(m + 2 * n), n:(2 * n)] = -np.eye(n)
        h = self.get('h') if self.get('h') is not None else np.empty(shape=(0,))
        h = np.append(h, np.append(x_init, -x_init))

        # Equality constraints
        A = np.pad(self['A'], [(0, 0), (0, n)]) if self.get('A') is not None else None

        lb = np.pad(self['lb'], (0, n)) if self.get('lb') is not None else None
        ub = np.pad(self['ub'], (0, n), constant_values=float('inf')) if self.get('ub') is not None else None

        # Override the original matrices
        self.update({'P': P,
                     'q': q,
                     'G': G,
                     'h': h,
                     'A': A,
                     'lb': lb,
                     'ub': ub})

        return None

    def is_feasible(self) -> bool:
        problem = qpsolvers.Problem(P=np.zeros(self.get('P').shape),
                                    q=np.zeros(self.get('P').shape[0]),
                                    G=self.get('G'),
                                    h=self.get('h'),
                                    A=self.get('A'),
                                    b=self.get('b'),
                                    lb=self.get('lb'),
                                    ub=self.get('ub'))

        # Convert to sparse matrices for best performance
        if self.solver in SPARSE_SOLVERS:
            if self['params'].get('sparse'):
                if problem.P is not None:
                    problem.P = scipy.sparse.csc_matrix(problem.P)
                if problem.A is not None:
                    problem.A = scipy.sparse.csc_matrix(problem.A)
                if problem.G is not None:
                    problem.G = scipy.sparse.csc_matrix(problem.G)
        solution = qpsolvers.solve_problem(problem=problem,
                                           solver=self.solver,
                                           initvals=self.get('x0'),
                                           verbose=False)
        return solution.found

    def solve(self) -> None:
        if self.solver in ['ecos', 'scs', 'clarabel']:
            if self.get('b').size == 1:
                self['b'] = np.array(self.get('b')).reshape(-1)

        P = self.get('P')
        if P is not None and not isPD(P):
            self['P'] = nearestPD(P)

        problem = qpsolvers.Problem(P=self.get('P'),
                                    q=self.get('q'),
                                    G=self.get('G'),
                                    h=self.get('h'),
                                    A=self.get('A'),
                                    b=self.get('b'),
                                    lb=self.get('lb'),
                                    ub=self.get('ub'))

        # Convert to sparse matrices for best performance
        if self.solver in SPARSE_SOLVERS:
            if self['params'].get('sparse'):
                if problem.P is not None:
                    problem.P = scipy.sparse.csc_matrix(problem.P)
                if problem.A is not None:
                    problem.A = scipy.sparse.csc_matrix(problem.A)
                if problem.G is not None:
                    problem.G = scipy.sparse.csc_matrix(problem.G)
        solution = qpsolvers.solve_problem(problem=problem,
                                           solver=self.solver,
                                           initvals=self.get('x0'),
                                           verbose=False)
        self['solution'] = solution
        return None

    # 0.5 * x' * P * x + q' * x + const
    def objective_value(self, x: np.ndarray, with_const: bool = True) -> float:
        const = 0 if self.get('constant') is None or not with_const else self['constant']
        return (0.5 * (x @ self.get('P') @ x) + self.get('q') @ x).item() + const

    def serialize(self, path, **kwargs):
        with open(path, 'wb') as f:
            pickle.dump(self, f, kwargs)

    @staticmethod
    def load(path, **kwargs):
        with open(path, 'rb'):
            return pickle.load(path, kwargs)
