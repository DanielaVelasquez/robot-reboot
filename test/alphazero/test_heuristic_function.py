import unittest
import numpy as np

from src.alphazero.heuristic_function import alpha_zero_heuristic_fn
from src.alphazero.state_statistics import StateStatistics


class TestHeuristicFunction(unittest.TestCase):

    def test_alpha_zero_heuristic_fn_increases_when_one_not_visited_same_predicted_probability(self):
        predicted_p = np.array([0.3, 0.3])

        statistics = StateStatistics(2)
        statistics.visit(0)
        statistics.visit(0)
        statistics.visit(0)
        statistics.add_value(0, 1)

        # p = [0.3, 0]
        # n = [3, 0]
        values = alpha_zero_heuristic_fn(predicted_p, statistics)
        self.assertTrue(values[1] > values[0])

    def test_alpha_zero_heuristic_fn_when_predicted_probabilities_different_statistics(self):
        predicted_p = np.array([-0.8, -0.9])

        statistics = StateStatistics(2)
        statistics.visit(0)
        statistics.add_value(0, 1)

        # p = [1, 0]
        # n = [1, 0]
        values = alpha_zero_heuristic_fn(predicted_p, statistics)
        self.assertTrue(values[1] > values[0])

    def test_alpha_zero_heuristic_fn_when_predicted_probabilities_same_statistics(self):
        predicted_p = np.array([1, 0])

        statistics = StateStatistics(2)
        statistics.visit(0)
        statistics.add_value(0, 1)

        # p = [1, 0]
        # n = [1, 0]
        values = alpha_zero_heuristic_fn(predicted_p, statistics)
        self.assertTrue(values[1] < values[0])

