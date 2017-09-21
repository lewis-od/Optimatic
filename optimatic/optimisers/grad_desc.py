"""
Gradient descent optimisation

Implements gradient descent to optimise a function
:math:`f:\mathbb{R}^n \\rightarrow \mathbb{R}`.

Uses the reccurence relation:

.. math::

    \mathbf{x}_n = \mathbf{x}_{n-1} - \gamma \\nabla f(\mathbf{x}_{n-1})
"""
import numpy as np
from optimatic.optimisers.optimiser_base import Optimiser as OptimiserBase

class Optimiser(OptimiserBase):
    """
    :param y: The function to optimise
    :param dy: The derivative of the function to optimise
    :param x0: The starting position for the algorithm
    :param precision: The precision to calculate the minimum to
    :param gamma: The starting value for :math:`\gamma`
    :param steps: The max number of iterations of the algorithm to run
    """
    def __init__(self, y, dy, x0, precision=0.0001, gamma=0.1, steps=10000):
        super(Optimiser, self).__init__(y, x0, precision=precision, steps=steps)
        self.dy = dy
        self.step_size = x0
        self.gamma = gamma

    def step(self):
        self.xn_1 = self.xn
        self.xn = self.xn_1 - self.gamma * self.dy(self.xn_1)

        grad_diff = self.dy(self.xn) - self.dy(self.xn_1)
        if grad_diff == 0.0:
            # Algorithm has converged
            return
        xs_diff = self.xn - self.xn_1
        self.gamma = np.dot(xs_diff, grad_diff)
        self.gamma /= np.linalg.norm(grad_diff) ** 2
