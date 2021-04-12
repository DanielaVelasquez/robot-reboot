import unittest

from src.mcts.state_statistics import StateStatistics
from src.uct.heuristic_function import uct_heuristic_fn


class TestHeuristicFunction(unittest.TestCase):

    def test_uct_heuristic_fn_when_visits_registered(self):
        statistics = StateStatistics(1)
        [statistics.visit(0) for _ in range(10)]
        [statistics.add_value(0, 1) for _ in range(6)]

        # p = [0.6]
        # n = [10]
        values = uct_heuristic_fn(statistics, 4)
        self.assertAlmostEqual(values[0], 0.748, delta=0.001)

    def test_uct_heuristic_fn_when_no_visits_registered(self):
        statistics = StateStatistics(1)
        values = uct_heuristic_fn(statistics, 4)
        self.assertAlmostEqual(values[0], 148, delta=0.001)
