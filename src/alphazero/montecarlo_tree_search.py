import numpy as np

from src.alphazero.state import State
from .game_player import GamePlayer
from .state_statistics import StateStatistics


class MonteCarloTreeSearch:
    """ Applies monte carlo tree search using a game player to find
    the best possible actions to take based on a state
    Attributes:
        heuristic_fn      (function):   function to evaluate how good is each action, receives probability and a state statistics
        max_depth         (number):     maximum depth for the tree while searching
        game_player       (GamePlayer): player for the game to optimize moves
        playouts          (number):     number of playouts per simulation
        states_statistics (dict):       dictionary between visited states and its statistics per action
    """

    def __init__(self, heuristic_fn, max_depth, game_player: GamePlayer, playouts=100):
        """
        Initializes a MonteCarlo tree search
        Args:
            heuristic_fn (function):   function to evaluate how good is each action, receives probability and a state statistics
            max_depth    (number):     maximum depth for the tree while searching
            game_player  (GamePlayer): player for the game to optimize moves
            playouts      (number):    number of playouts per simulation (default 100)
        """
        assert heuristic_fn is not None, "heuristic_fn must be provided"
        assert max_depth > 0, "Tree depth must be greater than 0"
        assert game_player is not None, "game_player must be provided"
        assert playouts > 0, "playouts must be greater than 0"

        self.__heuristic_fn = heuristic_fn
        self.__max_depth = max_depth
        self.__game_player = game_player
        self.__game = self.__game_player.game
        self.__playouts = playouts
        self.__states_statistics = {}

    @property
    def exploratory_parameter(self):
        return self.__exploratory_parameter

    @property
    def max_depth(self):
        return self.__max_depth

    @property
    def game_player(self):
        return self.__game_player

    @property
    def playouts(self):
        return self.__playouts

    def search(self, state: State):
        """Creates a tree with a series of simulated self-play games
        from root to leaf.
        Args:
            state (State): root state to start the tree
        Returns:
            p (numpy array): probability distribution of winning for each action
        """
        p = np.zeros(len(self.__game.actions), dtype=float)
        for i in range(len(self.__game.actions)):
            a = self.__game.actions[i]
            next_state = self.__game.apply(a, state)
            p[i] = self.__simulations(next_state, self.__playouts)
        return p

    def __simulations(self, state: State, n):
        """Self play n times starting at given state
        Args:
            state (State): state to start self-playing
            n     (int):   number of self- play games to perform
        Returns:
            v (float): average of values (wins, losses and draws) after self-playing from given state
        """
        v = np.array([self.__playout(state) for _ in range(n)])
        return v.mean()

    def __playout(self, state: State, depth=0):
        """Self plays the game from a given state.
        Selects next action to perform based on the probability calculated by the model. It plays until max depth is
        reached or the game is over.
        Args:
            state (State): state to start self-playing
            depth (int):   depth in the tree
        """
        if self.__game.is_over(state) or depth > self.__max_depth:
            return self.__game.get_value(state)
        p, v = self.__game_player.predict(state)

        state_stats = self.__get_state_statistics(state)
        heuristic_values = self.__heuristic_fn(p, state_stats)
        i_best = np.argsort(heuristic_values)[::-1][0]
        a = self.__game.actions[i_best]

        next_state = self.__game.apply(a, state)
        state_stats.visit(i_best)

        v = self.__playouts(next_state, depth + 1)
        state_stats.add_value(i_best, v)

        return v

    def __get_state_statistics(self, state):
        """Gets the statistics for a state from the dictionary states_statistics.
         If the state statistics don't exists, they are initiated with default values
        Args:
            state (State): state to get statistics for
        Returns:
            state_statistics (StateStatistics): statistics for the given state
        """
        if state not in self.__states_statistics:
            self.__states_statistics[state] = StateStatistics(len(self.__game.actions))
        return self.__states_statistics[state]
