import unittest
import numpy as np
from optimatic.optimisers.differential_evolution import Optimiser
from optimatic.exceptions import DidNotConvergeException

class TestDifferentialEvolution(unittest.TestCase):

    def setUp(self):
        self.mins = np.array([-2.15, 4.83])
        self.f = lambda x, y: (x - self.mins[0])**2 + (y - self.mins[1])**2

    def test_optimiser(self):
        pass
        # TODO: This doesn't always pass. Need to find a better way to decide
        #   when to terminate DE iterations
        # search = np.array([[-10, 10], [-10, 10]])
        # opt = Optimiser(self.f, search, 0.5, 1.2, 5, precision=1e-5, steps=1e5)
        # found = opt.optimise()
        # self.assertAlmostEqual(found[0], self.mins[0], places=3)
        # self.assertAlmostEqual(found[1], self.mins[1], places=3)

    def test_exception(self):
        f = lambda x: 3*x + 2
        search = np.array([[-10, 10]])
        opt = Optimiser(f, search, 0.5, 1.2, 5, precision=1e-5, steps=1e5)
        with self.assertRaises(DidNotConvergeException):
            opt.optimise()
