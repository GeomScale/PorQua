# GeoFin : a python library for portfolio optimization and index replication
# GeoFin is part of GeomScale project

# Copyright (c) 2024 Cyril Bachelard
# Copyright (c) 2024 Minh Ha Ho

# Licensed under GNU LGPL.3, see LICENCE file


import sys
import os
import unittest
from itertools import product
import pandas as pd
import numpy as np
import time
sys.path.insert(1, 'src')

from helper_functions import to_numpy
from data_loader import load_data_msci
from constraints import Constraints
from optimization import *
from optimization_data import OptimizationData



class TestQuadraticProgram(unittest.TestCase):

    def __init__(self, testname, universe = 'msci', solver_name = 'cvxopt', params = {}):
        super().__init__(testname)
        self._universe = universe
        self._solver_name = solver_name
        self.params = params

    @classmethod
    def setUpClass(cls):
        cls.data = load_data_msci(os.path.join(os.getcwd(), f'data{os.sep}'))

    def setUp(self):
        self.start_time = time.time()

    def constraints_from_params(self, selection):
        constraints = Constraints(selection=selection)

        if self.params.get('add_budget', False):
            constraints.add_budget()
        if self.params.get('add_box', None) is not None:
            constraints.add_box(self.params.get('add_box'))
        if self.params.get('add_ineq', False):
            linear_constraints = pd.DataFrame(np.random.rand(3, selection.size), columns=selection)
            sense = pd.Series(np.repeat('<=', 3))
            rhs = pd.Series(np.full(3, 0.5))
            constraints.add_linear(linear_constraints, None, sense, rhs, None)
        if self.params.get('add_l1', False):
            constraints.add_l1('turnover', rhs=1, x0=dict(zip(selection, np.zeros(selection.size))))

        return constraints

    def test_add_constraints(self):
        selection = self.data['return_series'].columns
        constraints = Constraints(selection = selection)

        constraints.add_budget()
        constraints.add_box("LongOnly")

        constraints.add_linear(None, pd.Series(np.random.rand(selection.size), index=selection), '<=', 1)
        constraints.add_linear(None, pd.Series(np.random.rand(selection.size), index=selection), '>=', -1)
        constraints.add_linear(None, pd.Series(np.random.rand(selection.size), index=selection), '=', 0.5)

        sub_universe = selection[:selection.size // 2]
        linear_constraints = pd.DataFrame(np.random.rand(3, sub_universe.size), columns=sub_universe)
        sense = pd.Series(np.repeat('=', 3))
        rhs = pd.Series(np.ones(3))
        constraints.add_linear(linear_constraints, None, sense, rhs, None)

        GhAb = constraints.to_GhAb()

        self.assertEqual(GhAb['G'].shape, (2, selection.size))
        self.assertEqual(GhAb['h'].shape, (2,))
        self.assertEqual(GhAb['A'].shape, (5, selection.size))
        self.assertEqual(GhAb['b'].shape, (5,))

        GhAb_with_box = constraints.to_GhAb(True)

        self.assertEqual(GhAb_with_box['G'].shape, (2 + 2 * selection.size, selection.size))
        self.assertEqual(GhAb_with_box['h'].shape, (2 + 2 * selection.size,))
        self.assertEqual(GhAb_with_box['A'].shape, (5, selection.size))
        self.assertEqual(GhAb_with_box['b'].shape, (5,))

# --------------------------------------------------------------------------
class TestLeastSquares(TestQuadraticProgram):

    def __init__(self, testname, universe='msci', solver_name='cvxopt', params={}):
        super().__init__(testname, universe, solver_name, params)

    def tearDown(self):
        self.run_time = time.time() - self.start_time
        recomputed = self.optim.model.objective_value(self.solution.x, False)
        print(f'{self._universe}-{self._solver_name}-{self.params}:\n\t* Solution found = {self.solution.found}\n\t* Utility = {recomputed}\n\t* Elapsed time: {self.run_time:.3f}(s)')

        if self.solution.found:
            from_solver = self.solution.obj
            if from_solver is not None:
                self.assertAlmostEqual(from_solver, recomputed)
        else:
            raise "Cannot find a solution"

    def prep_optim(self) -> Optimization:
        # optimization data
        return_series = self.data['return_series']
        bm_series = self.data['bm_series']
        optimization_data = OptimizationData(return_series=return_series, bm_series=bm_series, align=True)

        # Initialize optimization object
        optim = LeastSquares(solver_name = self._solver_name, sparse = True)

        # Set constraints
        selection = return_series.columns
        optim.params['l2_penalty'] = self.params.get('l2_penalty', 0)
        optim.constraints = self.constraints_from_params(selection)

        # Set objective
        optim.set_objective(optimization_data)
        optim.objective['q'] = to_numpy(optim.objective['q']) if 'q' in optim.objective.keys() else np.zeros(selection.size)

        # Initialize the optimization model
        optim.model_qpsolvers()
        return optim

    def test(self):
        self.optim = self.prep_optim()
        self.optim.solve()
        self.solution = self.optim.model['solution']


if __name__ == '__main__':
    suite = unittest.TestSuite()

    suite.addTest(TestQuadraticProgram('test_add_constraints'))

    universes = ['msci']
    solvers = ['highs', 'cvxopt']
    constraint_dict = {'l2_penalty': [0, 1],
                        'add_budget': [True, False],
                        'add_box': ["LongOnly", "LongShort", "Unbounded"],
                        'add_ineq': [True, False],
                        'add_l1': [True, False]}

    constraints_params = list(dict(zip(constraint_dict, x)) for x in product(*constraint_dict.values()))
    tests = product(universes, constraints_params, solvers)
    save_log = {universe : {} for universe in universes}

    for universe, params, solver in tests:
        suite.addTest(TestLeastSquares('test', universe, solver, params))

    runner = unittest.TextTestRunner()
    runner.run(suite)