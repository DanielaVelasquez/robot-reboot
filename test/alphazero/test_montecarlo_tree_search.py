import unittest

from unittest.mock import Mock
from src.alphazero.montecarlo_tree_search import MonteCarloTreeSearch


class TestMonteCarloTreeSearch(unittest.TestCase):
    def test_init_passes(self):
        mock_game_player = Mock()
        MonteCarloTreeSearch(0.1, 1, mock_game_player)

    def test_init_fails_when_exploratory_parameter_below_zero(self):
        mock_game_player = Mock()
        self.assertRaises(AssertionError, lambda: MonteCarloTreeSearch(-0.1, 1, mock_game_player))

    def test_init_fails_when_exploratory_parameter_above_one(self):
        mock_game_player = Mock()
        self.assertRaises(AssertionError, lambda: MonteCarloTreeSearch(1.1, 1, mock_game_player))

    def test_init_fails_when_max_depth_is_zero(self):
        mock_game_player = Mock()
        self.assertRaises(AssertionError, lambda: MonteCarloTreeSearch(0.1, 0, mock_game_player))

    def test_init_fails_when_max_depth_is_below_zero(self):
        mock_game_player = Mock()
        self.assertRaises(AssertionError, lambda: MonteCarloTreeSearch(0.1, -1, mock_game_player))

    def test_init_fails_when_game_player_none(self):
        self.assertRaises(AssertionError, lambda: MonteCarloTreeSearch(0.1, -1, None))
