"""
Numerical differentiation
"""
import numpy as np

def forward_diff(f, h, epsilon=1e-4):
    """
    Calculates the value of :math:`f^{\prime}(x=h)` using the foward difference
    method:

    .. math::

        f^{\prime}(h) \\approx \\frac{f(h + \epsilon) - f(h)}{\epsilon}

    :param f: The function to differentiate
    :param h: The value to evaluate the derivative at
    :param epsilon: The value to use for epsilon
    """
    numerator = f(h + epsilon) - f(h)
    df = numerator / epsilon
    return df

def central_diff(f, h, epsilon=1e-4):
    """
    Calculates the value of :math:`f^{\prime}(x=h)` using the central difference 
    method:

    .. math::

        f^{\prime}(h) \\approx \\frac{f(h + \epsilon) - f(h - \epsilon)}
            {2 \epsilon}

    :param f: The function to differentiate
    :param h: The value to evaluate the derivative at
    :param epsilon: The value to use for epsilon
    """
    numerator = f(h + epsilon) - f(h - epsilon)
    df = numerator / (2 * epsilon)
    return df
