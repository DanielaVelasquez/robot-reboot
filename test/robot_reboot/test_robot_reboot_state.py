import unittest
from unittest.mock import Mock
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

    def test_str(self):
        s = RobotRebootState(Mock(), [(0, 2), (4, 6)])
        self.assertEqual(str(s), '[(0, 2), (4, 6)]')
