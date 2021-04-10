from src.alphazero.state import State
from .game_player import GamePlayer
from ..mcts.monte_carlo_tree_search import MonteCarloTreeSearch


class AlphaZero(MonteCarloTreeSearch):

    def __init__(self, heuristic_fn, max_depth, game_player: GamePlayer, playouts=100):
        MonteCarloTreeSearch.__init__(self, heuristic_fn, max_depth, game_player, playouts)

    def _playout(self, state: State, depth=1):
        valid_actions = self._game.get_valid_actions(state)
        if self._game.is_over(state) or depth >= self._max_depth or len(valid_actions) == 0:
            return self._game.get_value(state)
        v, p = self._game_player.predict(state)

        state_stats = self._get_state_statistics(state)
        heuristic_values = self._heuristic_fn(p, state_stats)
        a, i_best = self._get_best_action(heuristic_values, valid_actions)

        next_state = self._game.apply(a, state)
        state_stats.visit(i_best)

        v = self._playout(next_state, depth + 1)
        state_stats.add_value(i_best, v)

        return v

