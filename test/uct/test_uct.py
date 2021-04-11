import unittest
from unittest.mock import Mock

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
