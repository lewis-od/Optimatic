from optimatic.grad_desc import Optimiser
import numpy as np

def f(x):
    return (x - 5.4) ** 2

def df(x):
    return 2 * (x - 5.4)

opt = Optimiser(f, df, 0.0)
