"""
Gradient descent optimisation

Implements gradient descent to optimise a function
:math:`f:\mathbb{R}^n \\rightarrow \mathbb{R}`.

Uses the reccurence relation:

.. math::

    \mathbf{x}_n = \mathbf{x}_{n-1} - \gamma_n \\nabla f(\mathbf{x}_{n-1})

Where

.. math::

    \gamma_n = \\frac{(\mathbf{x}_n - \mathbf{x}_{n-1}) \cdot \
        (\mathbf{\\nabla}f(\mathbf{x}_n) - \mathbf{\\nabla}\
        f(\mathbf{x}_{n-1}))}{||\mathbf{\\nabla}f(\mathbf{x}_n) - \
        \mathbf{\\nabla}f(\mathbf{x}_{n-1})||^2}
"""
import numpy as np
from optimatic.optimisers.optimiser_base import Optimiser as OptimiserBase
from optimatic.utils.differentiate import central_diff

class Optimiser(OptimiserBase):
    """
    :param f: The function to optimise
    :param x0: The starting position for the algorithm
    :param df: The derivative of the function to optimise. If this isn't
        provided, it will be estimated from :math:`f` using
        :func:`optimatic.utils.differentiate.central_diff`
    :param precision: The precision to calculate the minimum to
    :param gamma: The starting value for :math:`\gamma`
    :param steps: The max number of iterations of the algorithm to run
    """
    def __init__(self, f, x0, df=None, precision=0.0001, gamma=0.1,
        steps=10000):
        super(Optimiser, self).__init__(f, x0, precision=precision, steps=steps)
        if df is None:
            self.df = lambda x: central_diff(f, x)
        else:
            self.df = df
        self.step_size = x0
        self.gamma = gamma

    def step(self):
        self.xn_1 = self.xn
        self.xn = self.xn_1 - self.gamma * self.df(self.xn_1)

        grad_diff = self.df(self.xn) - self.df(self.xn_1)
        if grad_diff == 0.0:
            # Algorithm has converged
            return
        xs_diff = self.xn - self.xn_1
        self.gamma = np.dot(xs_diff, grad_diff)
        self.gamma /= np.linalg.norm(grad_diff) ** 2
