"""
Optimiser base class

All optimiser classes should inherit from this class
"""
from abc import ABCMeta, abstractmethod
from optimatic.exceptions import DidNotConvergeException
import numpy as np

class Optimiser(object):
    """
    :param f: The function to optimise
    :param x0: The starting position for the algorithm
    :param precision: The precision to calculate the minimum to
    :param steps: The max number of iterations of the algorithm to run
    """
    __metaclass__ = ABCMeta

    def __init__(self, f, x0, precision=1e-7, steps=10000):
        self.f = f
        self.xn = x0
        self.xn_1 = x0
        self.precision = precision
        self.steps = steps

    @abstractmethod
    def step(self):
        """Runs one iteration of the algorithm"""
        return

    def optimise(self):
        """Runs :func:`step` the specified number of times"""
        i = 0
        self.step()
        step_size = np.linalg.norm(self.xn - self.xn_1)
        while step_size > self.precision and i < self.steps:
            self.step()
            step_size = np.linalg.norm(self.xn - self.xn_1)
            i += 1
        if i >= self.steps and step_size > self.precision:
            msg = "Algorithm did not converge after {} steps.".format(i)
            raise DidNotConvergeException(msg)
        return self.xn
