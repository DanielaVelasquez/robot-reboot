from src.mcts.state_statistics import StateStatistics
import numpy as np

C = 0.4

def uct_heuristic_fn(predicted_p, statistics: StateStatistics):
    return statistics.p +  C * np.sqrt()
