from src.exceptions.exceptions import RequiredValueException
from src.exceptions.mcts.util import InvalidDepthException
from src.exceptions.util import assert_or_throw
from src.alphazero.model import Model
from src.game.state import State
from src.mcts.monte_carlo_tree_search import MonteCarloTreeSearch
from src.uct.heuristic_function import uct_heuristic_fn


class UCT(MonteCarloTreeSearch, Model):
    """Upper confidence bounds applied to trees. It computes a cheap hard-coded heuristic approximation
    to choose a move in a playout
    Attributes:
        heuristic_fn (function): function to evaluate the value of an action, receives the state statistics of the
                                 current state and the total number of games states simulated.
        max_depth     (number):  maximun depth for the three to search
    """

    def __init__(self, game, max_depth, heuristic_fn=uct_heuristic_fn, playouts=100):
        MonteCarloTreeSearch.__init__(self, game, playouts)
        Model.__init__(self, 'UCT Model', game)
        assert_or_throw(max_depth > 0, InvalidDepthException())
        assert_or_throw(heuristic_fn is not None, RequiredValueException("heuristic_fn"))
        self.__max_depth = max_depth
        self.__heuristic_fn = heuristic_fn

    @property
    def max_depth(self):
        return self.__max_depth

    @property
    def heuristic_fn(self):
        return self.__heuristic_fn

    def _playout(self, state: State, depth=1):
        valid_actions = self._game.get_valid_actions_next_state_map(state)
        if self._game.is_over(state) or depth >= self.__max_depth or len(valid_actions) == 0:
            return self._game.get_value(state)

        state_stats = self._get_state_statistics(state)
        heuristic_values = self.__heuristic_fn(state_stats, len(self._states_statistics))
        a, i_best = self._get_best_action(heuristic_values, valid_actions)

        next_state = self._game.apply(a, state)
        state_stats.visit(i_best)

        v = self._playout(next_state, depth + 1)
        state_stats.add_value(i_best, v)

        return v

    def predict(self, state: State):
        return self.game.get_value(state), self.search(state)
