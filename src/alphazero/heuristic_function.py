from src.alphazero.state_statistics import StateStatistics


def heuristic_fn(predicted_p, statistics: StateStatistics):
    visits = statistics.n.copy()
    visits[visits != 0] = 1 / visits[visits != 0]
    visits[visits == 0] = 2
    return predicted_p + visits + 0.0001 * statistics.p
