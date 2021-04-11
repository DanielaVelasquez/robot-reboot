import unittest
from unittest.mock import Mock

from exceptions.exceptions import RequiredValueException
from exceptions.mcts.monte_carlo_tree_search import InvalidPlayoutException
from exceptions.mcts.util import InvalidDepthException
from src.uct.uct import UCT


class TestUct(unittest.TestCase):
    def test_init_passes_without_default_parameters(self):
        mock_game = Mock()
        uct = UCT(mock_game, 1)
        self.assertEqual(uct.max_depth, 1)
        self.assertEqual(uct.game, mock_game)

    def test_init_with_default_parameters(self):
        mock_game = Mock()
        mock_heuristic_fn = Mock()
        uct = UCT(mock_game, 2, heuristic_fn=mock_heuristic_fn, playouts=100)
        self.assertEqual(uct.game, mock_game)
        self.assertEqual(uct.max_depth, 2)
        self.assertEqual(uct.heuristic_fn, mock_heuristic_fn)
        self.assertEqual(uct.playouts, 100)

    def test_init_throws_required_value_exception_when_game_is_none(self):
        self.assertRaises(RequiredValueException, lambda: UCT(None, 1))

    def test_init_throws_invalid_depth_exception_when_max_depth_is_zero(self):
        self.assertRaises(InvalidDepthException, lambda: UCT(Mock(), 0))

    def test_init_throws_invalid_depth_exception_when_max_depth_is_below_zero(self):
        self.assertRaises(InvalidDepthException, lambda: UCT(Mock(), -2))

    def test_init_throws_required_value_exception_when_heuristic_fn_is_none(self):
        self.assertRaises(RequiredValueException, lambda: UCT(Mock(), 2, heuristic_fn=None))

    def test_init_throws_invalid_playout_exception_when_playout_is_zero(self):
        self.assertRaises(InvalidPlayoutException, lambda: UCT(Mock(), 2, playouts=0))
