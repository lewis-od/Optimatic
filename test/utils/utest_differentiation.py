import unittest
import numpy as np
from optimatic.utils.differentiate import forward_diff, central_diff

class TestDifferentiation(unittest.TestCase):

    def setUp(self):
        self.f1 = lambda x: x**2
        self.f2 = np.sin

    def test_forward_diff(self):
        df1 = forward_diff(self.f1, 2)
        self.assertAlmostEqual(df1, 4, places=3)
        df2 = forward_diff(self.f2, np.pi)
        self.assertAlmostEqual(df2, -1, places=3)

    def test_central_diff(self):
        df1 = central_diff(self.f1, 2)
        self.assertAlmostEqual(df1, 4, places=4)
        df2 = central_diff(self.f2, np.pi)
        self.assertAlmostEqual(df2, -1, places=4)
