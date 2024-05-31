
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


from src.helper_functions import *
from src.covariance import Covariance
from src.optimization import *
from src.optimization_data import OptimizationData






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




def test_1():

    test = TestLeastSquares(universe = 'msci', solver_name = 'cvxopt')
    test.prep_optim()
    test.optim.solve()

    test.optim.model
    test.optim.results






