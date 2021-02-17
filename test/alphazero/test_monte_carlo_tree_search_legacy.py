import unittest

from unittest.mock import Mock
from src.alphazero.monte_carlo_tree_search_legacy import MonteCarloTreeSearchLegacy


class TestMonteCarloTreeSearchLegacy(unittest.TestCase):

    def test_playout_value_when_game_is_over(self):
        mock_game = Mock()
        mock_game.is_over.return_value = True
        mock_game.score.return_value = 20

        mtcs = MonteCarloTreeSearchLegacy(Mock(), 0.5, max_tree_depth=10)
        v = mtcs.playout_value(mock_game, 1)
        self.assertEqual(20, v)

    def test_playout_value_when_max_depth_reached(self):
        mock_game = Mock()
        mock_game.is_over.return_value = False
        mock_game.score.return_value = 30

        mtcs = MonteCarloTreeSearchLegacy(Mock(), 0.5, max_tree_depth=10)
        v = mtcs.playout_value(mock_game, 10)
        self.assertEqual(30, v)

    def test_get_best_action_with_one_action_when_next_state_never_visited(self):
        mock_neural_network = Mock()
        mock_neural_network.action_probabilities.return_value = {
            'a1': 1
        }
        mock_neural_network.heuristic_value.return_value = 0.8

        mock_game = Mock()
        mock_game.get_all_actions.return_value = ['a1']
        mock_game.get_valid_actions.return_value = ['a1']

        mtcs = MonteCarloTreeSearchLegacy(mock_neural_network, 0.5, max_tree_depth=10)
        action, value = mtcs.get_best_action(mock_game)
        self.assertEqual('a1', action)
        self.assertEqual(1.3, value)

    def test_get_best_action_with_one_action_when_next_state_visited(self):
        mock_neural_network = Mock()
        mock_neural_network.action_probabilities.return_value = {
            'a1': 1
        }
        mock_neural_network.heuristic_value.return_value = 0.8

        mock_game = Mock()
        mock_game.get_all_actions.return_value = ['a1']
        mock_game.get_valid_actions.return_value = ['a1']
        mock_game.observation.return_value = 'obs'

        mtcs = MonteCarloTreeSearchLegacy(mock_neural_network, 0.5, max_tree_depth=10)
        mtcs.three_visit_count = {hash('obs'): 3}
        action, value = mtcs.get_best_action(mock_game)
        self.assertEqual('a1', action)
        self.assertEqual(2.8, value)

    def test_get_best_action_with_two_actions_when_next_states_never_visited(self):
        mock_neural_network = Mock()
        mock_neural_network.action_probabilities.return_value = {
            'a1': 0.35,
            'a2': 0.65
        }
        mock_neural_network.heuristic_value.return_value = 0.8

        mock_game = Mock()
        mock_game.get_all_actions.return_value = ['a1', 'a2']
        mock_game.get_valid_actions.return_value = ['a1', 'a2']

        mtcs = MonteCarloTreeSearchLegacy(mock_neural_network, 0.5, max_tree_depth=10)
        action, value = mtcs.get_best_action(mock_game)
        self.assertEqual('a2', action)
        self.assertEqual(1.125, value)
