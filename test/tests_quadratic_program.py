
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
import pickle
import json
import time
sys.path.insert(1, '../src')

from helper_functions import *
from covariance import Covariance
from optimization import *
from optimization_data import OptimizationData



class TestQuadraticProgram(unittest.TestCase):

    def __init__(self, testname, universe = 'msci', solver_name = 'cvxopt'):
        super().__init__(testname)
        self._universe = universe
        self._solver_name = solver_name
        self.data = load_data(universe)

    def test_add_constraints(self):
        universe = self.data['X'].columns
        constraints = Constraints(selection = universe)

        constraints.add_budget()
        constraints.add_box("LongOnly")

        constraints.add_linear(None, pd.Series(np.random.rand(universe.size), index=universe), '<=', 1)
        constraints.add_linear(None, pd.Series(np.random.rand(universe.size), index=universe), '>=', -1)
        constraints.add_linear(None, pd.Series(np.random.rand(universe.size), index=universe), '=', 0.5)

        sub_universe = universe[:universe.size // 2]
        linear_constraints = pd.DataFrame(np.random.rand(3, sub_universe.size), columns=sub_universe)
        sense = pd.Series(np.repeat('=', 3))
        rhs = pd.Series(np.ones(3))
        constraints.add_linear(linear_constraints, None, sense, rhs, None)

        constraints.to_GhAb()
        constraints.to_GhAb(True)

# --------------------------------------------------------------------------
class SimpleLeastSquares(TestQuadraticProgram):

    def setUp(self):
        self.start_time = time.time()

    def tearDown(self):
        self.run_time = time.time() - self.start_time
        print('%s: Elapsed time: %.3f(s)' % (self.id(), self.run_time))
        serialize_solution(f'ls_{self._universe}_{self._solver_name}', self.solution, self.run_time)


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

    def test_least_square(self):
        self.prep_optim()
        self.optim.solve()
        self.solution = self.optim.model['solution']
        return None


if __name__ == '__main__':
    suite = unittest.TestSuite()

    suite.addTest(TestQuadraticProgram('test_add_constraints'))

    universes = ['msci', 'usa']
    solvers = list(set(qpsolvers.solvers.available_solvers) - {'gurobi', 'mosek', 'ecos', 'proxqp', 'piqp', 'scs'}) # FIXME: it depends on the installation?

    test_params = product(universes, solvers)
    save_log = {universe : {} for universe in universes}
    for universe, solver in test_params:
        suite.addTest(SimpleLeastSquares('test_least_square', universe, solver))

    runner = unittest.TextTestRunner()
    runner.run(suite)