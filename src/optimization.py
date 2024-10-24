'''
PorQua : a python library for portfolio optimization and backtesting
PorQua is part of GeomScale project

Copyright (c) 2024 Cyril Bachelard
Copyright (c) 2024 Minh Ha Ho

Licensed under GNU LGPL.3, see LICENCE file
'''


############################################################################
### OPTIMIZATION
############################################################################



from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


from helper_functions import to_numpy
from covariance import Covariance
from mean_estimation import MeanEstimator
from constraints import Constraints
from optimization_data import OptimizationData
import qp_problems

# https://github.com/qpsolvers/qpsolvers







class OptimizationParameter(dict):

    def __init__(self, **kwargs):
        super(OptimizationParameter, self).__init__(**kwargs)
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
                 **kwargs):
        self.params = OptimizationParameter(**kwargs) if params is None else params
        self.objective = Objective()
        self.constraints = Constraints() if constraints is None else constraints
        self.model = None
        self.results = None

    @abstractmethod
    def set_objective(self, optimization_data: OptimizationData) -> None:
        raise NotImplementedError("Method 'set_objective' must be implemented in derived class.")

    @abstractmethod
    def solve(self) -> bool:
        self.solve_qpsolvers()
        return self.results['status']

    def solve_qpsolvers(self) -> None:
        self.model_qpsolvers()
        self.model.solve()
        universe = self.constraints.selection
        solution = self.model['solution']
        status = solution.found
        weights = pd.Series(solution.x[:len(universe)] if status else [None] * len(universe),
                            index=universe)

        self.results = {'weights': weights.to_dict(),
                        'status': self.model['solution'].found}

        return None

    def model_qpsolvers(self) -> None:
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

        universe = self.constraints.selection

        # constraints
        constraints = self.constraints
        GhAb = constraints.to_GhAb()

        lb = constraints.box['lower'].to_numpy() if constraints.box['box_type'] != 'NA' else None
        ub = constraints.box['upper'].to_numpy() if constraints.box['box_type'] != 'NA' else None

        self.model = qp_problems.QuadraticProgram(P=self.objective['P'],
                                                  q=self.objective['q'],
                                                  constant=self.objective.get('constant'),
                                                  G=GhAb['G'],
                                                  h=GhAb['h'],
                                                  A=GhAb['A'],
                                                  b=GhAb['b'],
                                                  lb=lb,
                                                  ub=ub,
                                                  params=self.params)

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
            self.model.linearize_leverage_constraint(N=len(universe), leverage_budget=levcon['rhs'])
        return None


class EmptyOptimization(Optimization):

    def set_objective(self) -> None:
        pass

    def solve(self) -> bool:
        return super().solve()


class LeastSquares(Optimization):

    def __init__(self,
                 covariance: Optional[Covariance] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.covariance = covariance

    def set_objective(self, optimization_data: OptimizationData) -> None:

        X = optimization_data['return_series']
        y = optimization_data['bm_series']
        if self.params.get('log_transform'):
            X = np.log(1 + X)
            y = np.log(1 + y)

        # 0.5 * w * P * w' - q * w' + constant
        P = 2 * (X.T @ X)
        q = to_numpy(-2 * X.T @ y).reshape((-1,))
        constant = to_numpy(y.T @ y).item()

        l2_penalty = self.params.get('l2_penalty')
        if l2_penalty is not None and l2_penalty != 0:
            P += 2 * l2_penalty * np.eye(X.shape[1])

        self.objective = Objective(P=P,
                                   q=q,
                                   constant=constant)
        return None

    def solve(self) -> bool:
        return super().solve()


class WeightedLeastSquares(Optimization):

    def set_objective(self, optimization_data: OptimizationData) -> None:

        X = optimization_data['return_series']
        y = optimization_data['bm_series']
        if self.params.get('log_transform'):
            X = np.log(1 + X)
            y = np.log(1 + y)

        tau = self.params['tau']
        lambda_val = np.exp(-np.log(2) / tau)
        i = np.arange(X.shape[0])
        wt_tmp = lambda_val ** i
        wt = np.flip(wt_tmp / np.sum(wt_tmp) * len(wt_tmp))
        W = np.diag(wt)

        P = 2 * ((X.T).to_numpy() @ W @ X)
        q = -2 * (X.T).to_numpy() @ W @ y
        constant = (y.T).to_numpy() @ W @ y

        self.objective = Objective(P=P,
                                   q=q,
                                   constant=constant)
        return None

    def solve(self) -> bool:
        return super().solve()


class QEQW(Optimization):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.covariance = Covariance(method='duv')

    def set_objective(self, optimization_data: OptimizationData) -> None:
        X = optimization_data['return_series']
        covmat = self.covariance.estimate(X=X) * 2
        mu = np.zeros(X.shape[1])
        self.objective = Objective(P=covmat, q=mu)
        return None

    def solve(self) -> bool:
        return super().solve()


class LAD(Optimization):
    # Least Absolute Deviation (same as mean absolute deviation, MAD)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params['use_level'] = self.params.get('use_level', True)
        self.params['use_log'] = self.params.get('use_log', True)

    def set_objective(self, optimization_data: OptimizationData) -> None:
        X = optimization_data['return_series']
        y = optimization_data['bm_series']
        if self.params.get('use_level'):
            X = (1 + X).cumprod()
            y = (1 + y).cumprod()
            if self.params.get('use_log'):
                X = np.log(X)
                y = np.log(y)

        self.objective = Objective(X=X, y=y)

        return None

    def solve(self) -> bool:
        # Note: Should use an interior point linear solver instead of qpsolvers
        self.model_qpsolvers()
        self.model.solve()
        weights = pd.Series(self.model['solution'].x[0:len(self.constraints.selection)],
                            index=self.constraints.selection)
        self.results = {'weights': weights.to_dict()}
        return True

    def model_qpsolvers(self) -> None:
        # Data and constraints
        X = to_numpy(self.objective['X'])
        y = to_numpy(self.objective['y'])
        GhAb = self.constraints.to_GhAb()
        N = X.shape[1]
        T = X.shape[0]

        # Inequality constraints
        G_tilde = np.pad(GhAb['G'], [(0, 0), (0, 2 * T)]) if GhAb['G'] is not None else None
        h_tilde = GhAb['h']

        # Equality constraints
        A = GhAb['A']
        meq = 0 if A is None else 1 if A.ndim == 1 else A.shape[0]

        A_tilde = np.zeros(shape=(T, N + 2 * T)) if A is None else np.pad(A, [(0, T), (0, 2 * T)])
        A_tilde[meq:(T + meq), 0:N] = X
        A_tilde[meq:(T + meq), N:(N + T)] = np.eye(T)
        A_tilde[meq:(T + meq), (N + T):] = -np.eye(T)

        b_tilde = y if GhAb['b'] is None else np.append(GhAb['b'], y)

        lb = to_numpy(self.constraints.box['lower']) if self.constraints.box['box_type'] != 'NA' else np.full(N, -np.inf)
        lb = np.pad(lb, (0, 2 * T))

        ub = to_numpy(self.constraints.box['upper']) if self.constraints.box['box_type'] != 'NA' else np.full(N, np.inf)
        ub = np.pad(ub, (0, 2 * T), constant_values=np.inf)

        # Objective function
        q = np.append(np.zeros(N), np.ones(2 * T))
        P = np.diag(np.zeros(N + 2 * T))

        if 'leverage' in self.constraints.l1.keys():
            lev_budget = self.constraints.l1['leverage']['rhs']
            # Auxiliary variables to deal with the abs() function
            A_tilde = np.pad(A_tilde, [(0, 0), (0, 2 * N)])
            lev_eq = np.hstack((np.eye(N), np.zeros((N, 2 * T)), -np.eye(N), np.eye(N)))
            A_tilde = np.vstack((A_tilde, lev_eq))
            b_tilde = np.append(b_tilde, np.zeros())

            G_tilde = np.pad(G_tilde, [(0, 0), (0, 2 * N)])
            lev_ineq = np.append(np.zeros(N + 2 * T), np.ones(2 * N))
            G_tilde = np.vstack((G_tilde, lev_ineq))
            h_tilde = np.append(GhAb['h'], [lev_budget])

            lb = np.pad(lb, (0, 2 * N))
            ub = np.pad(lb, (0, 2 * N), constant_values=np.inf)

        self.model = qp_problems.QuadraticProgram(P=P,
                                                  q=q,
                                                  G=G_tilde,
                                                  h=h_tilde,
                                                  A=A_tilde,
                                                  b=b_tilde,
                                                  lb=lb,
                                                  ub=ub,
                                                  params=self.params)
        return None



class PercentilePortfolios(Optimization):

    def __init__(self, 
                 field: Optional[str] = None,
                 estimator: Optional[MeanEstimator] = None,
                 n_percentiles = 5,  # creates quintile portfolios by default.
                 **kwargs):
        super().__init__(**kwargs)
        self.estimator = estimator
        self.params = {'solver_name': 'percentile',
                       'n_percentiles': n_percentiles,
                       'field': field}

    def set_objective(self, optimization_data: OptimizationData) -> None:

        field = self.params.get('field')
        if self.estimator is not None:
            if field is not None:
                raise ValueError('Either specify a "field" or pass an "estimator", but not both.')
            else:
                scores = self.estimator.estimate(X = optimization_data['return_series'])
        else:
            if field is not None:
                scores = optimization_data['scores'][field]
            else:
                score_weights = self.params.get('score_weights')
                if score_weights is not None:
                    # Compute weighted average
                    scores = (
                        optimization_data['scores'][score_weights.keys()]
                        .multiply(score_weights.values())
                        .sum(axis=1)
                    )
                else:
                    scores = optimization_data['scores'].mean(axis = 1).squeeze()

        # Add tiny noise to zeros since otherwise there might be two threshold values == 0
        scores[scores == 0] = np.random.normal(0, 1e-10, scores[scores == 0].shape)
        self.objective = Objective(scores = -scores)

        return None

    def solve(self) -> bool:

        scores = self.objective['scores']
        N = self.params['n_percentiles']
        q_vec = np.linspace(0, 100, N + 1)
        th = np.percentile(scores, q_vec)
        lID = []
        w_dict = {}
        for i in range(1, len(th)):
            if i == 1:
                lID.append(list(scores.index[scores <= th[i]]))
            else:
                lID.append(list(scores.index[np.logical_and(scores > th[i-1], scores <= th[i])]))
            w_dict[i] = scores[lID[i-1]] * 0 + 1 / len(lID[i-1])     
        weights = scores * 0
        weights[w_dict[1].keys()] = 1 / len(w_dict[1].keys())
        weights[w_dict[N].keys()] = -1 / len(w_dict[N].keys())
        self.results = {'weights': weights.to_dict(),
                        'w_dict': w_dict}
        return True
