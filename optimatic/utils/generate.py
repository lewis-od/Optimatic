"""
Methods for generating data to fit using optimisation algorithms
"""
import numpy as np

def random_polynomial(degree, x, scale=5, noisy=True):
    """
    Generates a polynomial of the given degree with random coefficients, then
    adds some noise.

    :param degree: The degree of the polynomial to generate
    :param x: An array of x values to evaluate the polynomial :math:`y(x)` at
    :param scale: The stdev of the normal distribution from which the parameters
        wil be sampled
    :param noisy: Whether or not to add noise to the generated data. If true,
        noise will be sampled from a normal distribution with stdev scale/2
    :return y: The generated data points
    :return coefficients: The coefficients of the polynomial
    """
    degree += 1 # A 2nd degree polynomial has 3 parameters (a*x^2 + b*x + c)
    y = np.zeros(x.size)
    coefficients = np.random.normal(size=degree, scale=scale)
    for n, a in enumerate(coefficients):
        y += a*x**n
    if noisy:
        noise = np.random.normal(size=y.size, scale=scale/2)
        y += noise
    return y, coefficients
