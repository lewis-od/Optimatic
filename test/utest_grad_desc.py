import unittest
import numpy as np
from optimatic.optimisers.grad_desc import Optimiser

class TestGradientDescent(unittest.TestCase):

    def setUp(self):
        self.min = np.random.normal(scale=5)
        self.f = lambda x: (x - self.min)**2
        self.df = lambda x: 2 * (x - self.min)

    def test_optimiser(self):
        opt = Optimiser(self.f, self.df, np.random.normal(scale=5))
        found = opt.optimise()
        self.assertEqual(self.min, found)

if __name__ == '__main__':
    unittest.main()
