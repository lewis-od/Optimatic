Introduction
============
To use optimatic, first define the function you want to optimise:

.. code-block:: python

    def f(x):
        return (x - 2.4) ** 2

Some optimisation methods (e.g. gradient descent) also require you to define the
derivative of your function:

.. code-block:: python

    def df(x):
        return 2 * (x - 2.4)

Then import and initialise the optimiser you want to use, e.g.:

.. code-block:: python

    from optimatic.grad_desc import Optimiser
    opt = Optimiser(f, df, 6.0)

Then run either :code:`opt.step()` to run one step of the chosen optimisation
algorithm, or :code:`opt.optimise()` to run until either :code:`opt.steps` is
exceeded, or :code:`opt.precision` is met. See
:func:`~optimatic.grad_desc.Optimiser` for more details.
