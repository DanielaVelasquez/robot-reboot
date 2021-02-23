import unittest
from src.robot_reboot.robot_reboot_action import RobotRebootAction
from src.robot_reboot.util import Direction


class TestRobotRebootAction(unittest.TestCase):
    def test_init(self):
        a = RobotRebootAction(1, Direction.WEST)
        self.assertEqual(a.robot_id, 1)
        self.assertEqual(a.direction, Direction.WEST)

    def test_init_fails_when_robot_id_is_none(self):
        self.assertRaises(AssertionError, lambda : RobotRebootAction(None, Direction.WEST))

    def test_init_fails_when_direction_is_none(self):
        self.assertRaises(AssertionError, lambda : RobotRebootAction(1, None))
