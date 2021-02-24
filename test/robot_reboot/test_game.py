import unittest
from unittest.mock import Mock
import numpy as np

from src.robot_reboot.game import RobotRebootGame, Direction


class TestGame(unittest.TestCase):
    def test_init(self):
        maze = np.array([[0, 2, 0], [4, 5, 6], [0, 8, 0]])
        g = RobotRebootGame(2, maze)
        self.assertEqual(g.n_robots, 2)
        np.testing.assert_equal(g.maze, maze)
        directions = [d for d in Direction]
        self.assertEqual(len(g.actions), 8)

        self.assertEqual(g.actions[0].robot_id, 0)
        self.assertEqual(g.actions[0].direction, directions[0])

        self.assertEqual(g.actions[1].robot_id, 0)
        self.assertEqual(g.actions[1].direction, directions[1])

        self.assertEqual(g.actions[2].robot_id, 0)
        self.assertEqual(g.actions[2].direction, directions[2])

        self.assertEqual(g.actions[3].robot_id, 0)
        self.assertEqual(g.actions[3].direction, directions[3])

        self.assertEqual(g.actions[4].robot_id, 1)
        self.assertEqual(g.actions[4].direction, directions[0])

        self.assertEqual(g.actions[5].robot_id, 1)
        self.assertEqual(g.actions[5].direction, directions[1])

        self.assertEqual(g.actions[6].robot_id, 1)
        self.assertEqual(g.actions[6].direction, directions[2])

        self.assertEqual(g.actions[7].robot_id, 1)
        self.assertEqual(g.actions[7].direction, directions[3])

    def test_init_fails_when_n_robots_is_zero(self):
        self.assertRaises(AssertionError, lambda: RobotRebootGame(0, Mock()))

    def test_init_fails_when_n_robots_is_below_zero(self):
        self.assertRaises(AssertionError, lambda: RobotRebootGame(-1, Mock()))

    def test_init_fails_when_maze_has_walls_on_robot_cells(self):
        maze = np.array([[1, 2, 0], [4, 5, 6], [0, 8, 0]])
        self.assertRaises(AssertionError, lambda: RobotRebootGame(0, maze))
