import unittest
from unittest.mock import Mock
import numpy as np

from exceptions.exceptions import RequiredValueException
from exceptions.mcts.monte_carlo_tree_search import InvalidPlayoutException
from exceptions.mcts.util import InvalidDepthException
from src.mcts.state_statistics import StateStatistics
from src.robot_reboot.factory import RobotRebootFactory
from src.uct.heuristic_function import uct_heuristic_fn
from src.uct.uct import UCT
from test.alphazero.fake_data import FakeGame, FakeState
from test.mcts.util import assert_state


def fake_uct_heuristic_fn(state_statistics: StateStatistics, n):
    return state_statistics.p


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

    def test_search_with_depth_2_and_playouts_1(self):
        """
                        Tree built
                            s0

                (a1)    (a2)    (a3)    (a4)
                 s1      s2      s3      s1

                (a2)    (a3)            (a1)
                 s2      s3              s1

               v=-1      v=1     v=1     v=0

        """
        fake_game = FakeGame()
        fake_state = FakeState(fake_game, 0, 0)
        uct = UCT(fake_game, 2, heuristic_fn=fake_uct_heuristic_fn, playouts=1)
        p = uct.search(fake_state)
        np.testing.assert_equal(p, [-1, -1, 1, -1])
        assert_state(uct, 's1', n=[0, 0, 0, 1], w=[0, 0, 0, -1], p=[0, 0, 0, -1])
        assert_state(uct, 's2', n=[0, 0, 0, 1], w=[0, 0, 0, -1], p=[0, 0, 0, -1])
        assert_state(uct, 's4', n=[0, 0, 0, 1], w=[0, 0, 0, -1], p=[0, 0, 0, -1])

    def test_search_with_robot_reboot_game(self):
        np.random.seed(26)
        f = RobotRebootFactory()
        game, state, _ = f.create(31, locate_robot_close_goal=True, max_movements=3)
        uct = UCT(game, 5, heuristic_fn=uct_heuristic_fn, playouts=5)
        p = uct.search(state)
        np.testing.assert_equal(p, [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.2, 0.2, 0.2, 0.2])
        self.assertEqual(135, len(uct.states_statistics))
