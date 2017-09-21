"""
Implements Differential Evolution to find the minimum of a function
:math:`f:\mathbb{R}^n \\rightarrow \mathbb{R}`. See
https://en.wikipedia.org/wiki/Differential_evolution
"""

from optimatic.optimisers.optimiser_base import Optimiser as OptimiserBase
from copy import copy, deepcopy
import random
import numpy as np

class Optimiser(OptimiserBase):
    """
    :param f: The function being optimised
    :param search: The min/max value in each dimension to search in. This
        should be provided as a nummpy array of the format
        :code:`[[x_min, x_max], [y_min, y_max], ..]`, where
        :math:`[x_{min}, x_{max}]` is the search space in the :math:`x`
        dimension, etc. This should be of length :math:`n`.
    :param cr: Crossover probability
    :param F: Differential weight
    :param NP: Number of agents to use
    :param precision: Algorithm will stop when
        :math:`||\mathbf{x}_n - \mathbf{x}_{n-1}|| < \\text{precision}`, where
        :math:`\mathbf{x}_n` and :math:`\mathbf{x}_{n-1}` are the positions of
        the two agents closest to the minimum.
    :param steps: Max number of iterations to perform
    """
    def __init__(self, f, search, cr, F, NP, precision=1e-4, steps=10000):
        super(Optimiser, self).__init__(f, 0.0, precision=precision, steps=steps)

        if not search.shape[1] == 2:
            raise ValueError("Search array must be of shape (d,2)")
        elif not cr >= 0.0 and cr <= 1.0:
            raise ValueError("cr is a probability so must be between 0 and 1")
        elif not F >= 0.0 and F <= 2.0:
            raise ValueError("F must be between 0 and 2")
        elif not NP >= 4 and not NP == int(NP):
            raise ValueError("NP must be a integer between 0 and 4")

        self.dims = search.shape[0] # == len(search)
        self.search = search

        self.cr = cr
        self.F = F
        self.agents = [Agent(self.dims, search, self) for _ in range(NP)]

    def step(self):
        for index, agent in enumerate(self.agents):
            agents = copy(self.agents)
            agents.remove(agent)
            new_agent = deepcopy(agent)

            # Select 3 random agents
            selected = np.random.choice(agents, size=3, replace=False)

            # Calculate the new position of the agent based on the DE algorithm
            R = np.random.randint(0, high=self.dims+1)
            for i in range(self.dims):
                r_i = np.random.uniform(low=0, high=1)
                if r_i < self.cr or i == R:
                    new_pos = selected[0].position[i] + self.F * \
                        (selected[1].position[i] - selected[2].position[i])
                    # Ensure agents stay within the search space
                    if new_pos < self.search[i][0]:
                        new_pos = self.search[i][0]
                    elif new_pos > self.search[i][1]:
                        new_pos = self.search[i][1]
                    new_agent.position[i] = new_pos
                else:
                    new_agent.position[i] = agent.position[i]

            # Calculate the value of the function at the agent's new position
            new_value = self.f(*new_agent.position)
            if new_value < agent.value:
                # Update the agent if the new position is better than the old
                new_agent.value = new_value
                self.agents[index] = new_agent

        # Set the best 2 agents to x_n and x_{n-1}
        self.agents.sort(key=lambda a: a.value)
        self.xn = self.agents[0].position
        self.xn_1 = self.agents[1].position


class Agent(object):
    def __init__(self, dims, search, parent):
        self.position = np.zeros(dims)
        for i in range(dims):
            self.position[i] = np.random.uniform(*search[i])
        self.value = parent.f(*self.position)

    def __str__(self):
        return "({}): {}".format(self.position, self.value)
