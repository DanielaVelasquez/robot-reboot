from src.alphazero.state import State
from src.mcts.montecarlo_tree_search import MonteCarloTreeSearch


class AlphaZeroMCTS(MonteCarloTreeSearch):

    def __playout(self, state: State, depth=1):
        valid_actions = self.__game.get_valid_actions(state)
        if self.__game.is_over(state) or depth >= self.__max_depth or len(valid_actions) == 0:
            return self.__game.get_value(state)
        v, p = self.__game_player.predict(state)

        state_stats = self.__get_state_statistics(state)
        heuristic_values = self.__heuristic_fn(p, state_stats)
        a, i_best = self.__get_best_action(heuristic_values, valid_actions)

        next_state = self.__game.apply(a, state)
        state_stats.visit(i_best)

        v = self.__playout(next_state, depth + 1)
        state_stats.add_value(i_best, v)

        return v
