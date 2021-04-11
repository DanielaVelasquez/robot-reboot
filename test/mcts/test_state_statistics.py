import unittest
import numpy as np

from src.mcts.state_statistics import StateStatistics
from exceptions.mcts.state_statistics import EdgeNotVisitedException, InvalidNumberActionsException, \
    InvalidActionsTypeException


class TestStateStatistics(unittest.TestCase):

    def test_init(self):
        StateStatistics(1)

    def test_init_with_n_actions_zero(self):
        self.assertRaises(InvalidNumberActionsException, lambda: StateStatistics(0))

    def test_init_with_n_actions_below_zero(self):
        self.assertRaises(InvalidNumberActionsException, lambda: StateStatistics(-1))

    def test_init_with_n_actions_as_float(self):
        self.assertRaises(InvalidActionsTypeException, lambda: StateStatistics(1.2))

    def test_visit(self):
        obj = StateStatistics(2)
        obj.visit(0)
        self.assertEqual(1, obj.n[0])

    def test_add_value(self):
        obj = StateStatistics(2)
        obj.visit(0)
        obj.add_value(0, 1)
        obj.visit(0)
        obj.add_value(0, 0)
        np.testing.assert_equal([2, 0], obj.n)
        np.testing.assert_equal([0.5, 0], obj.p)
        np.testing.assert_equal([1, 0], obj.w)

    def test_add_value_when_no_visits_registered(self):
        obj = StateStatistics(2)
        self.assertRaises(EdgeNotVisitedException, lambda: obj.add_value(0, 1))
