from optimatic.optimisers.grad_desc import Optimiser
import numpy as np

minimum = np.random.normal(scale=5)
print("Actual minimum is: {}".format(minimum))

def f(x):
    return (x - minimum) ** 2

def df(x):
    return 2 * (x - minimum)

opt = Optimiser(f, df, np.random.normal(scale=5))
x = opt.optimise()

print("Calculated minimum is: {}".format(x))
