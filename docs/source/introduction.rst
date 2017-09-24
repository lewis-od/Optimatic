Introduction
============
To use optimatic, first define the function you want to optimise:

.. code-block:: python

    def f(x):
        return (x - 2.4) ** 2

Some optimisation methods (e.g. gradient descent) optionally allow you to
provide the derivative of your function:

.. code-block:: python

    def df(x):
        return 2 * (x - 2.4)

Then import and initialise the optimiser you want to use, e.g.:

.. code-block:: python

    from optimatic.grad_desc import Optimiser
    opt = Optimiser(f, df, np.array([6.0]))

All input values must be numpy arrays of the form :code:`[x, y, z, ...]`.

Now run either :code:`opt.step()` to run one step of the chosen optimisation
algorithm, or :code:`opt.optimise()` to run until or :code:`opt.precision` is
met. If the algorithm runs :code:`opt.steps` steps and still hasn't converged,
a :func:`~optimatic.exceptions.DidNotConvergeException` will be raised.

See :func:`~optimatic.optimisers.optimiser_base.Optimiser` for more details.
