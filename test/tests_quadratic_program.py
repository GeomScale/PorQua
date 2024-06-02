
############################################################################
### TESTS FOR CLASS QuadraticProgram
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     07.01.2024
# First version:    07.01.2024
# --------------------------------------------------------------------------



import unittest
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

    def __init__(self, universe = 'msci', solver_name = 'cvxopt'):
        self._testMethodName = 'Least squares'

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




def test_least_square():

    test = TestLeastSquares(universe = 'msci', solver_name = 'cvxopt')
    test.prep_optim()
    test.optim.solve()

    # test.optim.model
    # test.optim.results
    return test

def run_test(method, universe, solver):
    test = method(universe = universe, solver_name = solver)
    test.prep_optim()
    test.optim.solve()
    # print(f"- Solution is{'' if solution.is_optimal(1e-8) else ' NOT'} optimal")
    # print(f"- Primal residual: {solution.primal_residual():.1e}")
    # print(f"- Dual residual: {solution.dual_residual():.1e}")
    # print(f"- Duality gap: {solution.duality_gap():.1e}")
    return test

if __name__ == '__main__':
    # result = run_test(TestLeastSquares(), 'msci', 'cvxopt')
    result = test_least_square()
    print(result)




