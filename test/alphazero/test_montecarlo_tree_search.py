import unittest
from unittest.mock import Mock

import numpy as np

from src.alphazero.action import Action
from src.alphazero.game import Game
from src.alphazero.game_player import GamePlayer
from src.alphazero.montecarlo_tree_search import MonteCarloTreeSearch
from src.alphazero.state import State


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
        Game.__init__(self, [FakeAction(i + 1) for i in range(4)])

    def get_value(self, state: FakeState):
        if state.value == 3:
            return 1
        elif state.value == 1:
            return 0
        else:
            return -1

    def is_over(self, state: FakeState):
        return state.value == 3

    def get_score(self, state: FakeState):
        return state.sequence_i

    def apply(self, action: FakeAction, state: FakeState):
        """It goes to the state with the same action value
        E.g if action.value = 1 then it'll go to state.value =1 regardless of the starting state
        """
        return FakeState(state.sequence_i + 1, self, action.value)


class FakeGamePlayer(GamePlayer):

    def __init__(self):
        GamePlayer.__init__(self, None, FakeGame())

    def play(self, state: State):
        pass

    def predict(self, state: FakeState):
        actions_size = len(self.game.actions)
        p = np.zeros(actions_size, dtype=float)
        p[state.value % actions_size] = 1
        return p, np.nan


def assert_state(mcts, s, n=[], w=[], q=[]):
    np.testing.assert_equal(mcts.states_statistics[s].n, n)
    np.testing.assert_equal(mcts.states_statistics[s].w, w)
    np.testing.assert_equal(mcts.states_statistics[s].q, q)


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

    def test_search_with_depth_2(self):
        """
                        Tree built
                            s0

                (a1)    (a2)    (a3)    (a4)
                 s1      s2      s3      s1

                (a2)    (a3)            (a1)
                 s2      s3              s1

               v=-1      v=1     v=1     v=0

        """
        def heuristic_fn(p, _):
            return p

        fake_game_player = FakeGamePlayer()
        mcts = MonteCarloTreeSearch(heuristic_fn, 2, fake_game_player, playouts=1)
        fake_state = FakeState(0, fake_game_player.game, 0)
        p = mcts.search(fake_state)
        np.testing.assert_equal(p, [-1, 1, 1, 0])
        self.assertEqual(list(mcts.states_statistics.keys()), ['s1', 's2', 's4'])
        assert_state(mcts, 's1', n=[0, 1, 0, 0], w=[0, -1, 0, 0], q=[0, -1, 0, 0])
        assert_state(mcts, 's2', n=[0, 0, 1, 0], w=[0, 0, 1, 0], q=[0, 0, 1, 0])
        assert_state(mcts, 's4', n=[1, 0, 0, 0], w=[0, 0, 0, 0], q=[0, 0, 0, 0])

    def test_search_with_all_leaves_win(self):
        """
                                Tree built
                                    s0

                        (a1)    (a2)    (a3)    (a4)
                         s1      s2      s3      s1

                        (a2)    (a3)            (a1)
                         s2      s3              s1

                        (a3)                    (a2)
                         s3                      s2

                                                (a3)
                                                 s3
                        v=1      v=1     v=1     v=1

                """
        def heuristic_fn(p, _):
            return p

        fake_game_player = FakeGamePlayer()
        mcts = MonteCarloTreeSearch(heuristic_fn, 4, fake_game_player, playouts=1)
        fake_state = FakeState(0, fake_game_player.game, 0)
        p = mcts.search(fake_state)
        np.testing.assert_equal(p, [1, 1, 1, 1])
        self.assertEqual(list(mcts.states_statistics.keys()), ['s1', 's2', 's4'])
        assert_state(mcts, 's1', n=[0, 2, 0, 0], w=[0, 2, 0, 0], q=[0, 1, 0, 0])
        assert_state(mcts, 's2', n=[0, 0, 3, 0], w=[0, 0, 3, 0], q=[0, 0, 1, 0])
        assert_state(mcts, 's4', n=[1, 0, 0, 0], w=[1, 0, 0, 0], q=[1, 0, 0, 0])

    def test_search_clean_state_statistics_for_each_search(self):
        def heuristic_fn(p, _):
            return p

        fake_game_player = FakeGamePlayer()
        mcts = MonteCarloTreeSearch(heuristic_fn, 2, fake_game_player, playouts=1)
        fake_state = FakeState(0, fake_game_player.game, 0)
        p = mcts.search(fake_state)
        p = mcts.search(fake_state)
        np.testing.assert_equal(p, [-1, 1, 1, 0])
        self.assertEqual(list(mcts.states_statistics.keys()), ['s1', 's2', 's4'])
        assert_state(mcts, 's1', n=[0, 1, 0, 0], w=[0, -1, 0, 0], q=[0, -1, 0, 0])
        assert_state(mcts, 's2', n=[0, 0, 1, 0], w=[0, 0, 1, 0], q=[0, 0, 1, 0])
        assert_state(mcts, 's4', n=[1, 0, 0, 0], w=[0, 0, 0, 0], q=[0, 0, 0, 0])
