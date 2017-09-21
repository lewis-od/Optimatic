"""
Gradient descent optimisation

Implements gradient descent to optimise a function
:math:`f:\mathbb{R}^n \\rightarrow \mathbb{R}`.

Uses the reccurence relation:

.. math::

    \mathbf{x}_n = \mathbf{x}_{n-1} - \gamma \\nabla f(\mathbf{x}_{n-1})
"""
import numpy as np

class Optimiser(object):
    """
    :param y: The function to optimise
    :param dy: The derivative of the function to optimise
    :param x0: The starting position for the algorithm
    :param precision: The precision to calculate the minimum to
    :param gamma: The starting value for gamma
    :param steps: The max number of iterations of the algorithm to run
    """
    def __init__(self, y, dy, x0, precision=0.0001, gamma=0.1, steps=1000):
        self.y = y
        self.dy = dy
        self.precision = precision
        self.step_size = x0
        self.xn = x0
        self.xn_1 = x0
        self.gamma = gamma
        self.steps = steps

    def step(self):
        """Runs one iteration of the algorithm"""
        self.xn_1 = self.xn
        self.xn = self.xn_1 - self.gamma * self.dy(self.xn_1)

        grad_diff = self.dy(self.xn) - self.dy(self.xn_1)
        xs_diff = self.xn - self.xn_1
        self.gamma = np.dot(xs_diff, grad_diff)
        self.gamma /= np.linalg.norm(grad_diff) ** 2

    def optimise(self):
        """Runs :func:`step` the specified number of times"""
        i = 0
        self.step()
        step_size = np.linalg.norm(self.xn - self.xn_1)
        while step_size < self.precision and i < self.steps:
            self.step()
            step_size = np.linalg.norm(self.xn - self.xn_1)
            i += 1
        return self.xn
