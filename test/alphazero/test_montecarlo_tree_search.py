import unittest
from unittest.mock import Mock

import numpy as np

from exceptions.alphazero.monte_carlo_tree_search import InvalidDepthException, InvalidPlayoutException
from exceptions.exceptions import RequiredValueException
from src.alphazero.game_player import GamePlayer
from src.alphazero.montecarlo_tree_search import MonteCarloTreeSearch
from src.alphazero.state import State
from test.alphazero.fake_data import FakeGame, FakeState


class FakeGamePlayer(GamePlayer):

    def __init__(self, fn_predict_probability):
        GamePlayer.__init__(self, Mock(), FakeGame())
        self.__fn_predict_probability = fn_predict_probability

    def play(self, state: State):
        pass

    def predict(self, state: FakeState):
        return self.__fn_predict_probability(len(self.game.actions), state), np.nan


def assert_state(mcts, s, n=[], w=[], p=[]):
    np.testing.assert_equal(mcts.states_statistics[s].n, n)
    np.testing.assert_equal(mcts.states_statistics[s].w, w)
    np.testing.assert_equal(mcts.states_statistics[s].p, p)


def heuristic_fn(p, _):
    return p


def fn_predict_probability_1_for_next_action(actions_size, state:FakeState):
    """" Fake predict probabilities of winning over each action. Probabilities are set to 0 except for one action.
    E.g state = 1, actions_size = 3 then p = [0, 1, 0]
    E.g state = 3, actions_size = 3 then p = [1, 0, 0]
    (Function used in the FakeGamePlayer)
    Args:
        actions_size (int):   number of actions
        state        (State): state to calculate the probabilities of winning per actions
    Returns:
        p (np array): probability over actions. All actions with 0 except for one action with probability of 1
    """
    p = np.zeros(actions_size, dtype=float)
    p[state.value % actions_size] = 1
    return p


def fn_predict_probability_np_seed(actions_size, state):
    """Fake predict probabilities of winning over each action based on an uniform distribution
    with values between [-1, 1]
    (Function used in the FakeGamePlayer)
    Args:
       actions_size (int):   number of actions
        state       (State): state to calculate the probabilities of winning per actions
    Returns:
        p (np array): probality over actions with an uniform distribution.
    """
    return np.random.uniform(low=-1, high=1, size=actions_size)


class TestMonteCarloTreeSearch(unittest.TestCase):

    def test_init_passes_without_default_parameters(self):
        mock_game_player = Mock()
        MonteCarloTreeSearch(Mock(), 1, mock_game_player)

    def test_init_passes_with_default_parameters(self):
        mock_game_player = Mock()
        MonteCarloTreeSearch(Mock(), 1, mock_game_player, playouts=1)

    def test_init_fails_when_heuristic_fn_is_none(self):
        mock_game_player = Mock()
        self.assertRaises(RequiredValueException, lambda: MonteCarloTreeSearch(None, 1, mock_game_player))

    def test_init_fails_when_max_depth_is_zero(self):
        mock_game_player = Mock()
        self.assertRaises(InvalidDepthException, lambda: MonteCarloTreeSearch(Mock(), 0, mock_game_player))

    def test_init_fails_when_max_depth_is_below_zero(self):
        mock_game_player = Mock()
        self.assertRaises(InvalidDepthException, lambda: MonteCarloTreeSearch(Mock(), -1, mock_game_player))

    def test_init_fails_when_game_player_none(self):
        self.assertRaises(RequiredValueException, lambda: MonteCarloTreeSearch(Mock(), 1, None))

    def test_init_fails_when_playouts_is_zero(self):
        mock_game_player = Mock()
        self.assertRaises(InvalidPlayoutException, lambda: MonteCarloTreeSearch(Mock(), 1, mock_game_player, playouts=0))

    def test_init_fails_when_playouts_is_below_zero(self):
        mock_game_player = Mock()
        self.assertRaises(InvalidPlayoutException, lambda: MonteCarloTreeSearch(Mock(), 1, mock_game_player, playouts=-1))

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
        fake_game_player = FakeGamePlayer(fn_predict_probability_1_for_next_action)
        mcts = MonteCarloTreeSearch(heuristic_fn, 2, fake_game_player, playouts=1)
        fake_state = FakeState(0, fake_game_player.game, 0)
        p = mcts.search(fake_state)
        np.testing.assert_equal(p, [-1, 1, 1, 0])
        self.assertEqual(list(mcts.states_statistics.keys()), ['s1', 's2', 's4'])
        assert_state(mcts, 's1', n=[0, 1, 0, 0], w=[0, -1, 0, 0], p=[0, -1, 0, 0])
        assert_state(mcts, 's2', n=[0, 0, 1, 0], w=[0, 0, 1, 0], p=[0, 0, 1, 0])
        assert_state(mcts, 's4', n=[1, 0, 0, 0], w=[0, 0, 0, 0], p=[0, 0, 0, 0])

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

        fake_game_player = FakeGamePlayer(fn_predict_probability_1_for_next_action)
        mcts = MonteCarloTreeSearch(heuristic_fn, 4, fake_game_player, playouts=1)
        fake_state = FakeState(0, fake_game_player.game, 0)
        p = mcts.search(fake_state)
        np.testing.assert_equal(p, [1, 1, 1, 1])
        self.assertEqual(list(mcts.states_statistics.keys()), ['s1', 's2', 's4'])
        assert_state(mcts, 's1', n=[0, 2, 0, 0], w=[0, 2, 0, 0], p=[0, 1, 0, 0])
        assert_state(mcts, 's2', n=[0, 0, 3, 0], w=[0, 0, 3, 0], p=[0, 0, 1, 0])
        assert_state(mcts, 's4', n=[1, 0, 0, 0], w=[1, 0, 0, 0], p=[1, 0, 0, 0])

    def test_search_clean_state_statistics_for_each_search(self):
        fake_game_player = FakeGamePlayer(fn_predict_probability_1_for_next_action)
        mcts = MonteCarloTreeSearch(heuristic_fn, 2, fake_game_player, playouts=1)
        fake_state = FakeState(0, fake_game_player.game, 0)
        p = mcts.search(fake_state)
        p = mcts.search(fake_state)
        np.testing.assert_equal(p, [-1, 1, 1, 0])
        self.assertEqual(list(mcts.states_statistics.keys()), ['s1', 's2', 's4'])
        assert_state(mcts, 's1', n=[0, 1, 0, 0], w=[0, -1, 0, 0], p=[0, -1, 0, 0])
        assert_state(mcts, 's2', n=[0, 0, 1, 0], w=[0, 0, 1, 0], p=[0, 0, 1, 0])
        assert_state(mcts, 's4', n=[1, 0, 0, 0], w=[0, 0, 0, 0], p=[0, 0, 0, 0])

    def test_search_with_random_probabilities(self):
        """
                                                    Tree built
                                                        s0

                    (a1)                        (a2)            (a3)                    (a4)
                     s1                          s2              s3                     s4

            (a4)    (a1)    (a1)        (a1)    (a1)    (a2)                    (a1)    (a4)    (a1)
             s4      s1      s1          s1      s1      s2                     s1      s4      s1

            (a1)    (a2)    (a3)        (a4)    (a3)    (a1)                    (a3)   (a4)    (a3)
             s1      s2      s3          s4      s3      s1                      s3     s4      s3

            v=0      v=-1    v=1         v=-1    v=1     v=0    v=1 v=1 v=1      v=1    v=-1    v=0

        """
        np.random.seed(26)
        fake_game_player = FakeGamePlayer(fn_predict_probability_np_seed)
        mcts = MonteCarloTreeSearch(heuristic_fn, 3, fake_game_player, playouts=3)
        fake_state = FakeState(0, fake_game_player.game, 0)
        p = mcts.search(fake_state)
        np .testing.assert_equal(p, [0, 0, 1, 0])
        self.assertEqual(sorted(mcts.states_statistics.keys()), sorted(['s1', 's2', 's4']))
        assert_state(mcts, 's1', n=[3, 1, 3, 2], w=[0, -1, 3, -1], p=[0, -1, 1, -0.5])
        assert_state(mcts, 's2', n=[3, 1, 0, 0], w=[0, 0, 0, 0], p=[0, 0, 0, 0])
        assert_state(mcts, 's4', n=[3, 0, 0, 2], w=[1, 0, 0, -2], p=[1/3, 0, 0, -1])
