import numpy as np


class StateStatistics:
    """
    Statistics for each action for a state visited while building a Monte Carlo Tree search,

    Attributes:
        n (numpy array): Total visits for each action
        w (numpy array): Sum value(wins, lost and draws) over the leaf states of simulations for each action
        q (numpy array): Probability of winning for each action
    """

    def __init__(self, n_actions):
        """Initializes statistics for a state
        Args:
            n_actions (number): number of actions
        """
        self.__n = np.zeros(n_actions, dtype=float)
        self.__w = np.zeros(n_actions, dtype=float)
        self.__q = np.zeros(n_actions, dtype=float)

    @property
    def n(self):
        return self.__n

    @property
    def w(self):
        return self.__w

    @property
    def q(self):
        return self.__q
