from abc import ABC, abstractmethod

import numpy as np

from exceptions.mcts.monte_carlo_tree_search import InvalidPlayoutException
from exceptions.util import assertOrThrow
from src.alphazero.game import Game
from src.alphazero.state import State
from src.alphazero.state_statistics import StateStatistics


class MonteCarloTreeSearch(ABC):
    """ Applies monte carlo tree search using a game player to find
    the best possible actions to take based on a state
    Attributes:
        game              (Game):     game used for the search
        playouts          (number):   number of playouts per simulation
        states_statistics (dict):     dictionary between visited states and its statistics per action
    """

    def __init__(self, game: Game, playouts=100):
        """
        Initializes a MonteCarlo tree search
        Args:
            game      (Game):  game used for the search
            playouts (number): number of playouts per simulation (default 100)
        """
        assertOrThrow(playouts > 0, InvalidPlayoutException())
        self._game = game
        self._playouts = playouts
        self._states_statistics = {}

    @property
    def game(self):
        return self._game

    @property
    def playouts(self):
        return self._playouts

    @property
    def states_statistics(self):
        return self._states_statistics

    def search(self, state: State):
        """Creates a tree with a series of simulated self-play games
        from root to leaf.
        Args:
            state (State): root state to start the tree
        Returns:
            p (numpy array): probability distribution of winning for each action
        """
        self._states_statistics = {}
        p = np.zeros(len(self._game.actions), dtype=float)
        for i in range(len(self._game.actions)):
            a = self._game.actions[i]
            if a in self._game.get_valid_actions(state):
                next_state = self._game.apply(a, state)
                p[i] = self._simulations(next_state, self._playouts)
        return p

    def _simulations(self, state: State, n):
        """Self play n times starting at given state
        Args:
            state (State): state to start self-playing
            n     (int):   number of self- play games to perform
        Returns:
            v (float): average of values (wins, losses and draws) after self-playing from given state
        """
        v = np.array([self._playout(state) for _ in range(n)])
        return v.mean()

    def _get_best_action(self, heuristic_values, valid_actions):
        """Gets the best next action to execute based on the results from the heuristic values
        and valid actions from the current state
        Args:
            heuristic_values (np array): heuristic values for each action
            valid_actions (list): actions that will alter the current state
        """
        ordered_actions = np.argsort(heuristic_values)[::-1]
        i = 0
        i_best = ordered_actions[i]
        a = self._game.actions[i_best]
        while a not in valid_actions:
            i += 1
            i_best = ordered_actions[i]
            a = self._game.actions[i_best]
        return a, i_best

    def _get_state_statistics(self, state):
        """Gets the statistics for a state from the dictionary.
         If the state statistics don't exists, they are initiated with default values
        Args:
            state (State): state to get statistics for
        Returns:
            state_statistics (StateStatistics): statistics for the state
        """
        s = str(state)
        if s not in self._states_statistics:
            self._states_statistics[s] = StateStatistics(len(self._game.actions))
        return self._states_statistics[s]

    @abstractmethod
    def _playout(self, state: State, depth=1):
        """Self plays the game from a given state.
        Selects next action based on the heuristic function value for each action. The heuristic function utilizes
        the probability calculated by the model and the states statistics. It plays until max depth is
        reached or the game is over.
        Args:
            state (State): state to start self-playing
            depth (int):   depth in the tree
        """
        pass