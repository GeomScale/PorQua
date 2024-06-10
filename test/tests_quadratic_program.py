
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

        GhAb = constraints.to_GhAb()

        self.assertEqual(GhAb['G'].shape, (2, universe.size))
        self.assertEqual(GhAb['h'].shape, (2,))
        self.assertEqual(GhAb['A'].shape, (5, universe.size))
        self.assertEqual(GhAb['b'].shape, (5,))

        GhAb_with_box = constraints.to_GhAb(True)

        self.assertEqual(GhAb_with_box['G'].shape, (2 + 2 * universe.size, universe.size))
        self.assertEqual(GhAb_with_box['h'].shape, (2 + 2 * universe.size,))
        self.assertEqual(GhAb_with_box['A'].shape, (5, universe.size))
        self.assertEqual(GhAb_with_box['b'].shape, (5,))

# --------------------------------------------------------------------------
class SimpleLeastSquares(TestQuadraticProgram):

    def setUp(self):
        self.start_time = time.time()

    def tearDown(self):
        self.run_time = time.time() - self.start_time
        print(f'{self.id()}-{self._universe}-{self._solver_name}: Elapsed time: {self.run_time:.3f}(s)')
        # serialize_solution(f'ls_{self._universe}_{self._solver_name}', self.solution, self.run_time)


    def prep_optim(self, constraints, rebdate: str = None) -> None:
        # Initialize optimization object
        optim = LeastSquares(solver_name = self._solver_name)

        # Add constraints
        optim.constraints = constraints

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

    def least_square(self):
        constraints = Constraints(selection = self.data['X'].columns)
        constraints.add_box(box_type = 'LongOnly')
        constraints.add_budget()

        self.prep_optim(constraints)
        self.optim.solve()
        self.solution = self.optim.model['solution']
        return None

    def least_square_with_inequalities(self):
        universe = self.data['X'].columns
        constraints = Constraints(selection = universe)
        constraints.add_budget()

        linear_constraints = pd.DataFrame(np.random.rand(3, universe.size), columns=universe)
        sense = pd.Series(np.repeat('<=', 3))
        rhs = pd.Series(np.ones(3))
        constraints.add_linear(linear_constraints, None, sense, rhs, None)

        self.prep_optim(constraints)
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
        suite.addTest(SimpleLeastSquares('least_square', universe, solver))
        suite.addTest(SimpleLeastSquares('least_square_with_inequalities', universe, solver))

    runner = unittest.TextTestRunner()
    runner.run(suite)