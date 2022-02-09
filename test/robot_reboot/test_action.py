import unittest
from src.robot_reboot.action import RobotRebootAction
from src.exceptions.exceptions import RequiredValueException
from src.robot_reboot.util import Direction


class TestRobotRebootAction(unittest.TestCase):
    def test_init(self):
        a = RobotRebootAction(1, Direction.WEST)
        self.assertEqual(a.robot_id, 1)
        self.assertEqual(a.direction, Direction.WEST)

    def test_init_fails_when_robot_id_is_none(self):
        self.assertRaises(RequiredValueException, lambda: RobotRebootAction(None, Direction.WEST))

    def test_init_fails_when_direction_is_none(self):
        self.assertRaises(RequiredValueException, lambda: RobotRebootAction(1, None))
