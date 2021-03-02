import unittest
import numpy as np
from unittest.mock import Mock, PropertyMock
from src.robot_reboot.state import RobotRebootState


class TestRobotRebootState(unittest.TestCase):

    def test_init_with_sequence_i_sets_all_given_values(self):
        s = RobotRebootState(Mock(), [(0, 2), (4, 6)], sequence_i=3)
        self.assertEqual(s.sequence_i, 3)
        self.assertEqual(s.robots_positions, [(0, 2), (4, 6)])
        self.assertIsNotNone(s.game)

    def test_init_without_sequence_i_sets_sequence_i_to_zero(self):
        s = RobotRebootState(Mock(), [(0, 2), (4, 6)])
        self.assertEqual(s.sequence_i, 0)

    def test_init_fails_when_empty_list_robot_positions(self):
        self.assertRaises(AssertionError, lambda: RobotRebootState(Mock(), []))

    def test_init_fails_when_sequence_i_below_zero(self):
        self.assertRaises(AssertionError, lambda: RobotRebootState(Mock(), [(0, 2), (4, 6)], sequence_i=-1))

    def test_str(self):
        s = RobotRebootState(Mock(), [(0, 2), (4, 6)])
        self.assertEqual(str(s), '[(0, 2), (4, 6)]')

    def test_is_robot__on_returns_true_when_robot_on_position(self):
        s = RobotRebootState(Mock(), [(0, 2), (4, 6)])
        self.assertTrue(s.is_robot_on((4, 6)))

    def test_is_robot__on_returns_false_when_no_robot_on_position(self):
        s = RobotRebootState(Mock(), [(0, 2), (4, 6)])
        self.assertFalse(s.is_robot_on((5, 6)))

    def test_init_fails_when_one_robot_position_with_negative_value(self):
        self.assertRaises(AssertionError, lambda: RobotRebootState(Mock(), [(0, -1), (4, 6)]))

    def test_init_fails_when_one_robot_x_position_out_of_maze_bounds(self):
        mock_game = Mock()
        mock_maze_property = PropertyMock(return_value=np.array([[0 for i in range(3)] for j in range(3)]))
        type(mock_game).maze = mock_maze_property
        self.assertRaises(AssertionError, lambda: RobotRebootState(mock_game, [(0, 2), (6, 0)]))

    def test_init_fails_when_one_robot_out_of_maze_bounds(self):
        mock_game = Mock()
        mock_maze_property = PropertyMock(return_value=np.array([[0 for i in range(3)] for j in range(3)]))
        type(mock_game).maze = mock_maze_property
        self.assertRaises(AssertionError, lambda: RobotRebootState(mock_game, [(0, 2), (6, 6)]))
