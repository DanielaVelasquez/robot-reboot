import numpy as np

from src.alphazero.state import State
from .game_player import GamePlayer


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
        p = np.empty(len(self.__game.actions), dtype=float)
        for i in range(len(self.__game.actions)):
            a = self.__game.actions[i]
            next_state = self.__game.apply(a, state)
            p[i] = self.__simulations(next_state)
        return p

    def __simulations(self, state: State, n):
        v = np.array([self.__playout(state) for _ in range(n)])
        return v.mean()

    def __playout(self, state: State, depth=0):
        if self.__game.is_over(state) or depth > self.__max_depth:
            return self.__game.get_value(state)
        p, v = self.__game_player.predict(state)

        stats = self.__states_statictis[state]
        heuristic_values = self.__heuristic_fn(p, stats)
        i_best = np.argsort(heuristic_values)[::-1][0]
        a = self.__game.actions[i_best]

        next_state = self.__game.apply(a, state)
        v = self.__playouts(next_state, depth + 1)
        return v
