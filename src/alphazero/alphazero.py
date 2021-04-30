from exceptions.mcts.util import InvalidDepthException
from exceptions.exceptions import RequiredValueException
from exceptions.util import assertOrThrow
from src.game.state import State
from .game_player import GamePlayer
from .heuristic_function import alpha_zero_heuristic_fn
from .model import Model
from ..mcts.monte_carlo_tree_search import MonteCarloTreeSearch


class AlphaZero(MonteCarloTreeSearch, Model):
    """AlphaZero algorithm, it uses MCTS and a neural network to lead the search
    Attributes:
        heuristic_fn      (function):   function to evaluate how good is each action, receives probability and a state statistics
        max_depth         (number):     maximum depth for the tree while searching
        game_player       (GamePlayer): player for the game to optimize moves
    """

    def __init__(self, max_depth, game_player: GamePlayer, heuristic_fn=alpha_zero_heuristic_fn, playouts=100):
        assertOrThrow(game_player is not None, RequiredValueException("game_player"))
        MonteCarloTreeSearch.__init__(self, game_player.game, playouts=playouts)
        assertOrThrow(heuristic_fn is not None, RequiredValueException("heuristic_fn"))
        assertOrThrow(max_depth > 0, InvalidDepthException())

        self.__heuristic_fn = heuristic_fn
        self.__max_depth = max_depth
        self.__game_player = game_player

    @property
    def max_depth(self):
        return self.__max_depth

    @property
    def game_player(self):
        return self.__game_player

    @property
    def heuristic_fn(self):
        return self.__heuristic_fn

    def _playout(self, state: State, depth=1):
        valid_actions = self._game.get_valid_actions(state)
        if self._game.is_over(state) or depth >= self.__max_depth or len(valid_actions) == 0:
            return self._game.get_value(state)
        v, p = self.__game_player.predict(state)

        state_stats = self._get_state_statistics(state)
        heuristic_values = self.__heuristic_fn(p, state_stats, len(self._states_statistics))
        a, i_best = self._get_best_action(heuristic_values, valid_actions)

        next_state = self._game.apply(a, state)
        state_stats.visit(i_best)

        v = self._playout(next_state, depth + 1)
        state_stats.add_value(i_best, v)

        return v

    def predict(self, state: State):
        return self.game.get_value(state), self.search(state)
