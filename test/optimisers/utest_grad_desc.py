import unittest
import numpy as np
from optimatic.optimisers.grad_desc import Optimiser
from optimatic.exceptions import DidNotConvergeException

class TestGradientDescent(unittest.TestCase):

    def setUp(self):
        self.min = np.random.normal(scale=5)
        self.f = lambda x: (x - self.min)**2
        self.df = lambda x: 2 * (x - self.min)

    def test_optimiser(self):
        x0 = np.array([np.random.normal(scale=5)])
        opt = Optimiser(self.f, x0, df=self.df)
        found = opt.optimise()
        self.assertEqual(self.min, found)

    def test_optimiser_no_derivative(self):
        x0 = np.array([np.random.normal(scale=5)])
        opt = Optimiser(self.f, x0)
        found = opt.optimise()
        self.assertEqual(self.min, found)

    def test_error(self):
        f = lambda x: 3*x + 2
        df = lambda x: 3
        opt = Optimiser(f, np.random.normal(scale=5), df=df)
        with self.assertRaises(DidNotConvergeException):
            opt.optimise()

if __name__ == '__main__':
    unittest.main()
