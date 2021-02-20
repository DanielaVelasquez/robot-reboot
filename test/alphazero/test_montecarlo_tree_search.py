import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, PropertyMock
from src.alphazero.montecarlo_tree_search import MonteCarloTreeSearch


class TestMonteCarloTreeSearch(unittest.TestCase):

    def test_init_passes_without_default_parameters(self):
        mock_game_player = Mock()
        MonteCarloTreeSearch(Mock(), 1, mock_game_player)

    def test_init_passes_with_default_parameters(self):
        mock_game_player = Mock()
        MonteCarloTreeSearch(Mock(), 1, mock_game_player, playouts=1)

    def test_init_fails_when_max_depth_is_zero(self):
        mock_game_player = Mock()
        self.assertRaises(AssertionError, lambda: MonteCarloTreeSearch(Mock(), 0, mock_game_player))

    def test_init_fails_when_max_depth_is_below_zero(self):
        mock_game_player = Mock()
        self.assertRaises(AssertionError, lambda: MonteCarloTreeSearch(Mock(), -1, mock_game_player))

    def test_init_fails_when_game_player_none(self):
        self.assertRaises(AssertionError, lambda: MonteCarloTreeSearch(Mock(), -1, None))

    def test_init_fails_when_playouts_is_zero(self):
        mock_game_player = Mock()
        self.assertRaises(AssertionError, lambda: MonteCarloTreeSearch(Mock(), 1, mock_game_player, playouts=0))

    def test_init_fails_when_playouts_is_below_zero(self):
        mock_game_player = Mock()
        self.assertRaises(AssertionError, lambda: MonteCarloTreeSearch(Mock(), -1, mock_game_player, playouts=-1))

    def test_search(self):
        mock_heuristic_fn = Mock()
        """"
        mock.side_effect = [5, 4, 3, 2, 1]
        mock(), mock(), mock() //(5, 4, 3)
        """
        mock_game_player = Mock()
        mock_game = Mock()
        mock_state = MagicMock()

        type(mock_game_player).game = PropertyMock(return_value=mock_game)
        type(mock_game).actions = PropertyMock(return_value=[f'a{i}' for i in range(10)])
        mock_state.__str__.return_value = 's0'

        mcts = MonteCarloTreeSearch(mock_heuristic_fn, 2, mock_game_player, playouts=2)
        p = mcts.search(mock_state)
