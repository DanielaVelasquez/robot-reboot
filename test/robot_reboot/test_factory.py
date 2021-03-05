import unittest

from exceptions.robot_reboot.factory import UnsupportedMazeSize
from src.robot_reboot.factory import RobotRebootFactory


class TestRobotRebootFactory(unittest.TestCase):
    def test_create_fails_when_unsupported_maze_size(self):
        factory = RobotRebootFactory()
        self.assertRaises(UnsupportedMazeSize, lambda: factory.create(3))
