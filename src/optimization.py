# GeoFin : a python library for portfolio optimization and index replication
# GeoFin is part of GeomScale project

# Copyright (c) 2024 Cyril Bachelard
# Copyright (c) 2024 Minh Ha Ho

# Licensed under GNU LGPL.3, see LICENCE file


from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import gurobipy as gp

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
            print(f'transaction_cost = {transaction_cost}')
            self.model.linearize_turnover_objective(pd.Series(x_init), transaction_cost)

        # Turnover constraint
        if tocon and not transaction_cost and x_init is not None:
            self.model.linearize_turnover_constraint(pd.Series(x_init), tocon['rhs'])

        # Leverage constraint
        levcon = self.constraints.l1.get('leverage')
        if levcon is not None:
            self.model.linearize_leverage_constraint(N=len(universe), leverage_budget=levcon['rhs'])
        return None


class LeastSquares(Optimization):

    def __init__(self,
                 covariance: Optional[Covariance] = None,
                 *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        self.covariance = covariance

    def set_objective(self, optimization_data: OptimizationData) -> None:
        X, y = optimization_data.view(self.constraints.selection, mode='log')

        # 0.5 * w * P * w' - q * w' + constant
        P = 2 * (X.T @ X)
        q = to_numpy(-2 * X.T @ y).reshape((-1,))
        constant = to_numpy(y.T @ y).item()

        l2_penalty = self.params.get('l2_penalty')
        if l2_penalty is not None and l2_penalty != 0:
            P += 2 * l2_penalty * np.eye(X.shape[1])

        self.objective = Objective(P=P, q=q, constant=constant)
        return None

    def solve(self) -> bool:
        return super().solve()


class WeightedLeastSquares(Optimization):

    def set_objective(self, optimization_data: OptimizationData) -> None:
        X, y = optimization_data.view(self.constraints.selection, mode='log')

        tau = self.params['tau']
        lambda_val = np.exp(-np.log(2) / tau)
        i = np.arange(X.shape[0])
        wt_tmp = lambda_val ** i
        wt = np.flip(wt_tmp / np.sum(wt_tmp) * len(wt_tmp))
        W = np.diag(wt)

        P = 2 * ((X.T).to_numpy() @ W @ X)
        q = -2 * (X.T).to_numpy() @ W @ y
        constant = (y.T).to_numpy() @ W @ y

        self.objective = Objective(P=P, q=q, constant=constant)
        return None

    def solve(self) -> bool:
        return super().solve()


class QEQW(Optimization):

    def __init__(self, *arg, **kwarg):
        covariance = Covariance(method='duv')
        super().__init__(covariance=covariance, *arg, **kwarg)

    def set_objective(self, optimization_data: OptimizationData) -> None:
        covmat = self.covariance.estimate(X=optimization_data['X']) * 2
        mu = np.zeros(optimization_data['X'].shape[1])
        self.objective = Objective(P=covmat, q=mu)
        return None

    def solve(self) -> bool:
        return super().solve()


class LAD(Optimization):
    # Least Absolute Deviation (same as mean absolute deviation, MAD)

    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        self.params['use_level'] = self.params.get('use_level', True)
        self.params['use_log'] = self.params.get('use_log', True)

    def set_objective(self, optimization_data: OptimizationData) -> None:
        # X = np.log(1 + training_data['X_train'])
        # y = np.log(1 + training_data['X_bm'])
        # X = training_data['X_train']
        # y = training_data['X_bm'].squeeze()
        # if self.params.get('use_level'):
        #     X = (1 + X).cumprod()
        #     y = (1 + y).cumprod()
        #     if self.params.get('use_log'):
        #         X = np.log(X)
        #         y = np.log(y)
        X = np.log(1 + optimization_data['X'])
        y = np.log(1 + optimization_data['y'])
        self.objective = Objective(X=X, y=y)

        return None

    def solve(self) -> None:
        solver_name = self.params['solver_name']
        if solver_name == 'gurobi':
            self.solve_gurobi()
        else:
            self.solve_qpsolvers()
        return None

    def solve_gurobi(self) -> None:
        # Data and constraints
        X = to_numpy(self.objective['X'])
        y = to_numpy(self.objective['y'])
        GhAb = self.constraints.to_GhAb()
        N = X.shape[1]
        T = X.shape[0]

        # Initiate an empty model
        self.model = gp.Model('portfolio')
        self.model.Params.LogToConsole = 0

        # Add matrix variable for the asset weights
        lb = to_numpy(self.constraints.box['lower'])
        ub = to_numpy(self.constraints.box['upper'])
        x = self.model.addMVar(N, lb=lb, ub=ub, vtype=gp.GRB.CONTINUOUS, name='x')

        # Auxiliary variables to deal with the abs() function
        aux_lad_pos = self.model.addMVar(T, lb=np.zeros(T), name='aux_lad_pos')
        aux_lad_neg = self.model.addMVar(T, lb=np.zeros(T), name='aux_lad_neg')
        self.model.addConstr(X @ x + aux_lad_pos - aux_lad_neg == y, name='aux_lad')

        # Objective function
        self.model.setObjective(aux_lad_pos.sum() + aux_lad_neg.sum(), gp.GRB.MINIMIZE)

        # Add linear inequality constraints
        if GhAb['G'] is not None:
            self.model.addConstr(GhAb['G'] @ x <= GhAb['h'], name='Gh')

        # Add linear equality constraints
        if GhAb['A'] is not None:
            self.model.addConstr(GhAb['A'] @ x == GhAb['b'], name='Ab')

        # Add quadratic inequality constraints
        quadcon = self.constraints.quadratic
        if quadcon:
            for key, value in quadcon.items():
                if isinstance(value, dict):
                    self.model.addConstr(value['q'].T @ x + (x @ value['Qc'] @ x) <= value['rhs'], key)

        # Leverage constraint
        if 'leverage' in self.constraints.l1.keys():
            levcon = self.constraints.l1['leverage']
            # Auxiliary variables to deal with the abs() function
            aux_lvrg_pos = self.model.addMVar(N, lb=np.zeros(N), name='aux_lvrg_pos')
            aux_lvrg_neg = self.model.addMVar(N, lb=np.zeros(N), name='aux_lvrg_neg')
            self.model.addConstr(aux_lvrg_pos - aux_lvrg_neg == x, name='aux_leverage')
            self.model.addConstr(aux_lvrg_pos.sum() + aux_lvrg_neg.sum() <= levcon['rhs'], 'leverage_budget')

        # Solve and store results
        self.model.optimize()
        solved = True if self.model.status == 2 or (
                    self.model.status == 13 and self.params['allow_suboptimal']) else False

        if solved:
            weights = pd.Series(self.model.x[0:N], index=self.constraints.selection)
            obj_val = self.model.objVal
        else:
            weights = pd.Series(np.nan, index=self.constraints.selection)
            obj_val = np.nan
        self.results = {'weights': weights.to_dict(),
                        'objective': obj_val,
                        'status': self.model.status}
        self.model.dispose()
        return None

    def solve_qpsolvers(self) -> None:
        # Note: Should use an interior point linear solver instead of qpsolvers
        self.model_qpsolvers()
        self.model.solve()
        weights = pd.Series(self.model['solution'].x[0:len(self.constraints.selection)],
                            index=self.constraints.selection)
        self.results = {'weights': weights.to_dict()}
        return None

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

        lb = to_numpy(self.constraints.box['lower']) if self.constraints.box['box_type'] != 'NA' else np.full(N,
                                                                                                              -np.inf)
        lb = np.pad(lb, (0, 2 * T))

        ub = to_numpy(self.constraints.box['upper']) if self.constraints.box['box_type'] != 'NA' else np.full(N, np.inf)
        ub = np.pad(lb, (0, 2 * T), constant_values=np.inf)

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
