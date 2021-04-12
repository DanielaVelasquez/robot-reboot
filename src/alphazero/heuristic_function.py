from src.mcts.state_statistics import StateStatistics
from src.uct.heuristic_function import uct_heuristic_fn


def alpha_zero_heuristic_fn(predicted_p, statistics: StateStatistics, N):
    return predicted_p + 0.35 * uct_heuristic_fn(statistics, N)
