import numpy as np

from exceptions.alphazero.state_statistics import EdgeNotVisitedException, InvalidNumberActionsException, \
    InvalidActionsTypeException
from exceptions.util import assertOrThrow


class StateStatistics:
    """
    Statistics for each action for a state visited while building a Monte Carlo Tree search,

    Attributes:
        n (numpy array): Total visits for each action
        w (numpy array): Sum values(wins, lost and draws) over the leaf states of simulations for each action
        p (numpy array): Probability of winning for each action
    """

    def __init__(self, n_actions):
        """Initializes statistics for a state
        Args:
            n_actions (number): number of actions
        """
        assertOrThrow(n_actions > 0, InvalidNumberActionsException())
        assertOrThrow(type(n_actions) is int, InvalidActionsTypeException())

        self.__n = np.zeros(n_actions, dtype=float)
        self.__w = np.zeros(n_actions, dtype=float)
        self.__p = np.zeros(n_actions, dtype=float)

    @property
    def n(self):
        return self.__n

    @property
    def w(self):
        return self.__w

    @property
    def p(self):
        return self.__p

    def visit(self, action_i):
        """Register a visit from a state taking one action
        Args:
            action_i (int): index of the action taken
        """
        self.__n[action_i] += 1

    def add_value(self, action_i, v):
        """Adds the value (wins, lost and draws) from a leaf state
        Args:
            action_i (int): index of the action taken
            v        (int): value of the leaf state (1 = win, 0 = draw, -1 = lost)
        """
        assertOrThrow(self.__n[action_i] != 0, EdgeNotVisitedException(action_i))
        self.__w[action_i] += v
        self.__p[action_i] = self.__w[action_i] / self.__n[action_i]

    def __str__(self):
        return f'n = {self.__n}\nw = {self.__w}\np = {self.__p}'
