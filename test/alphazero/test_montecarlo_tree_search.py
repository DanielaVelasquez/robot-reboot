import unittest

from unittest.mock import Mock
from numpy import np

from src.alphazero.action import Action
from src.alphazero.model import Model
from src.alphazero.monte_carlo_tree_search_legacy import MonteCarloTreeSearchLegacy
from src.alphazero.game import Game
from src.alphazero.state import State
from src.alphazero.game_player import GamePlayer


class FakeAction(Action):

    def __init__(self, value):
        Action.__init__(self)
        self.__value = value

    @property
    def value(self):
        return self.__value

    def __str__(self):
        return f'a{self.__value}'


class FakeState(State):

    def __init__(self, sequence_i, game, value):
        State.__init__(self, sequence_i, game)
        self.__value = value

    @property
    def value(self):
        return self.__value

    def __str__(self):
        return f's{self.__value}'


class FakeGame(Game):
    def __init__(self):
        Game.__init__(self, [FakeAction(i) for i in range(1, 3)])

    def get_value(self, state: FakeState):
        if state.value == 3:
            return 1
        elif state.value == 2:
            return 0
        else:
            return -1

    def is_over(self, state: FakeState):
        return state.value == 3 or state.value == 2

    def get_score(self, state: FakeState):
        return state.sequence_i

    def apply(self, action: FakeAction, state: FakeState):
        """It goes to the state with the same action value
        E.g if action.value = 1 then it'll go to state.value =1 regardless of the starting state
        """
        return FakeState(state.sequence_i + 1, self, action.value)


class FakeGamePlayer(GamePlayer):

    def __init__(self, game: FakeGame):
        GamePlayer.__init__(self, None, game)

    def play(self, state: State):
        pass

    def predict(self, state: FakeState):
        actions_size = len(self.game.actions)
        p = np.zeros(actions_size, dtype=float)
        p[state.value % actions_size] = 1
        return p, np.nan


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
