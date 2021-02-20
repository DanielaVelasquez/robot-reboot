import unittest

from unittest.mock import Mock
from numpy import np

from src.alphazero.action import Action
from src.alphazero.model import Model
from src.alphazero.monte_carlo_tree_search_legacy import MonteCarloTreeSearchLegacy
from src.alphazero.game import Game
from src.alphazero.state import State
from src.alphazero.game_player import GamePlayer


class TestAction(Action):

    def __init__(self, value):
        Action.__init__(self)
        self.__value = value

    @property
    def value(self):
        return self.__value

    def __str__(self):
        return f'a{self.__value}'


class TestState(State):

    def __init__(self, sequence_i, game, value):
        State.__init__(self, sequence_i, game)
        self.__value = value

    @property
    def value(self):
        return self.__value

    def __str__(self):
        return f's{self.__value}'


class TestGame(Game):
    def __init__(self):
        Game.__init__(self, [TestAction(i) for i in range(1, 3)])

    def get_value(self, state: TestState):
        if state.value == 3:
            return 1
        elif state.value == 2:
            return 0
        else:
            return -1

    def is_over(self, state: TestState):
        return state.value == 3 or state.value == 2

    def get_score(self, state: TestState):
        return state.sequence_i

    def apply(self, action: TestAction, state: TestState):
        """It goes to the state with the same action value
        E.g if action.value = 1 then it'll go to state.value =1 regardless of the starting state
        """
        return TestState(state.sequence_i + 1, self, action.value)


class TestGamePlayer(GamePlayer):

    def __init__(self, game: TestGame):
        GamePlayer.__init__(self, None, game)

    def play(self, state: State):
        pass

    def predict(self, state:TestState):
        actions_size = len(self.game.actions)
        p = np.zeros(actions_size, dtype=float)
        p[state.value % actions_size] = 1
        return p, np.nan


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
