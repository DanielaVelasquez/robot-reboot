import unittest
import numpy as np

from src.alphazero.heuristic_function import alpha_zero_heuristic_fn
from src.mcts.state_statistics import StateStatistics


class TestHeuristicFunction(unittest.TestCase):

    def test_alpha_zero_heuristic_fn_when_visits_registered(self):
        predicted_p = np.array([0.3])

        statistics = StateStatistics(1)
        [statistics.visit(0) for _ in range(10)]
        [statistics.add_value(0, 1) for _ in range(6)]

        # p = [0.6]
        # n = [10]
        values = alpha_zero_heuristic_fn(predicted_p, statistics, 4)
        self.assertAlmostEqual(values[0], 0.5621, delta=0.001)

    def test_alpha_zero_heuristic_fn_when_no_visits_registered(self):
        predicted_p = np.array([0.3])
        statistics = StateStatistics(1)
        values = alpha_zero_heuristic_fn(predicted_p, statistics, 4)
        self.assertAlmostEqual(values[0], 52.426, delta=0.001)
