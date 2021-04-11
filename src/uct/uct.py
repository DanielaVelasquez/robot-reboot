from exceptions.mcts.util import InvalidDepthException
from exceptions.util import assertOrThrow
from src.game.state import State
from src.mcts.monte_carlo_tree_search import MonteCarloTreeSearch
from src.uct.heuristic_function import uct_heuristic_fn


class UCT(MonteCarloTreeSearch):
    """Upper confidence bounds applied to trees. It computes a cheap hard-coded heuristic approximation
    to choose a move in a playout
    Attributes:
        heuristic_fn (function): function to evaluate the value of an action, receives the state statistics of the
                                 current state and the total number of games states simulated.
        max_depth     (number):  maximun depth for the three to search
    """

    def __init__(self, game, max_depth, heuristic_fn=uct_heuristic_fn, playouts=100):
        MonteCarloTreeSearch.__init__(self, game, playouts)
        assertOrThrow(max_depth > 0, InvalidDepthException())
        self.__max_depth = max_depth
        self.__heuristic_fn = heuristic_fn

    @property
    def max_depth(self):
        return self.__max_depth

    @property
    def heuristic_fn(self):
        return self.__heuristic_fn

    def _playout(self, state: State, depth=1):
        pass
