
############################################################################
### TESTS FOR CLASS QuadraticProgram
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     07.01.2024
# First version:    07.01.2024
# --------------------------------------------------------------------------



import unittest
from itertools import product
import pandas as pd
import numpy as np
import scipy
import qpsolvers
from qpsolvers import solve_qp
from typing import Tuple
import sys
sys.path.insert(1, '../src')

from helper_functions import *
from covariance import Covariance
from optimization import *
from optimization_data import OptimizationData



class TestQuadraticProgram(unittest.TestCase):

    def __init__(self, universe = 'msci', solver_name = 'cvxopt'):
        self._universe = universe
        self._solver_name = solver_name
        self.load_data(universe = universe)

    def load_data(self, universe = 'msci') -> None:
        if universe == 'msci':
            self.data = load_data_msci()
        elif universe == 'usa':
            self.data = load_data_usa()
        else:
            raise ValueError('Universe not recognized.')
        return None



# --------------------------------------------------------------------------
class TestLeastSquares(TestQuadraticProgram):

    def __init__(self, name_suffix, universe, solver_name):
        super().__init__(universe, solver_name)
        self._testMethodName = f'LeastSquare_{name_suffix}'

    def prep_optim(self, rebdate: str = None) -> None:
        # Initialize optimization object
        optim = LeastSquares(solver_name = self._solver_name)

        # Add constraints
        optim.constraints = Constraints(selection = self.data['X'].columns)
        optim.constraints.add_box(box_type = 'LongOnly')
        optim.constraints.add_budget()

        # Set objective
        optimization_data = OptimizationData(X = self.data['X'], y = self.data['y'], align = True)
        optim.set_objective(optimization_data = optimization_data)

        # Ensure that P and q are numpy arrays
        if 'P' in optim.objective.keys():
            if hasattr(optim.objective['P'], "to_numpy"):
                optim.objective['P'] = optim.objective['P'].to_numpy()
        else:
            raise ValueError("Missing matrix 'P' in objective.")
        if 'q' in optim.objective.keys():
            if hasattr(optim.objective['q'], "to_numpy"):
                optim.objective['q'] = optim.objective['q'].to_numpy()
        else:
            optim.objective['q'] = np.zeros(len(optim.constraints.selection))

        # Initialize the optimization model
        optim.model_qpsolvers()

        # Attach the optimization object to self
        self.optim = optim

        return None

def test_least_square(params):
    universe = params[0]
    solver = params[1]
    name_suffix = f'_{universe}_{solver}'

    test = TestLeastSquares(name_suffix, universe, solver)
    test.prep_optim()
    test.optim.solve()

    return test

def run_test(method, universe, solver):
    test = method(universe = universe, solver_name = solver)
    test.prep_optim()
    return test

if __name__ == '__main__':
    universe_set = ['msci']
    solver_names = ['cvxopt', 'qpalm']

    test_params = product(universe_set, solver_names)

    for i, params in enumerate(test_params):
        test_obj = test_least_square(params)
        solution = test_obj.optim.model['solution']

        print(f"- Primal objective at the solution is {solution.obj}")
        print(f"- Solution is {solution.x} and {'' if solution.is_optimal(1e-6) else ' NOT'} optimal")
        print(f"- Solution is{'' if solution.is_optimal(1e-8) else ' NOT'} optimal")
        print(f"- Value of the primal objective at the solution is {solution.obj}")
        print(f"- Primal residual: {solution.primal_residual():.1e}")
        print(f"- Dual residual: {solution.dual_residual():.1e}")
        print(f"- Duality gap: {solution.duality_gap()}")

