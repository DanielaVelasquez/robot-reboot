import numpy as np

from src.mcts.state_statistics import StateStatistics


def uct_heuristic_fn(statistics: StateStatistics, N, C=0.4):
    """Heuristic value of a move for the UCT algorithm
    Args:
        statistics (StateStatistics) : statistics of a state
        N (float): Total number of game states simulated
        C (float): Adjustable parameter for exploration
    """
    n_i = statistics.n.copy()
    n_i[n_i == 0] = 1e-5
    return statistics.p + C * np.sqrt(np.log(N)/n_i)
