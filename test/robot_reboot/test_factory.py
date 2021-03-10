import unittest
import numpy as np

from exceptions.robot_reboot.factory import UnsupportedMazeSize
from src.robot_reboot.factory import RobotRebootFactory
from src.robot_reboot.util import join_quadrants


class TestRobotRebootFactory(unittest.TestCase):
    def test_create_fails_when_unsupported_maze_size(self):
        factory = RobotRebootFactory()
        self.assertRaises(UnsupportedMazeSize, lambda: factory.create(3))

    def test_create_returns_game_when_n_is_31(self):
        factory = RobotRebootFactory()
        game, state, selected_quadrants = factory.create(31)
        confs, n_robots = factory.get_game_configurations(31)
        expected_maze = join_quadrants(confs[selected_quadrants[0]].quadrant, confs[selected_quadrants[1]].quadrant,
                                       confs[selected_quadrants[2]].quadrant, confs[selected_quadrants[3]].quadrant)
        np.testing.assert_equal(expected_maze, game.maze)
        self.assertEqual(n_robots, len(state.robots_positions))
        self.assertTrue(game.goal_house.house not in state.robots_positions)

    def test_create_different_mazes_every_time(self):
        factory = RobotRebootFactory(26)
        game_1, state_1, selected_quadrants_1 = factory.create(31)
        game_2, state_2, selected_quadrants_2 = factory.create(31)
        self.assertFalse(np.array_equal(game_1.maze, game_2.maze))
        self.assertNotEqual(state_1.robots_positions, state_2.robots_positions)
        self.assertFalse(np.array_equal(selected_quadrants_1, selected_quadrants_2))
