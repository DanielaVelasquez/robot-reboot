import unittest

import numpy as np

from src.encoders.maze_and_robot_positioning_encoder import MazeAndRobotPositioningEncoder
from src.robot_reboot.action import RobotRebootAction
from src.robot_reboot.direction import Direction
from src.robot_reboot.state import RobotRebootState
from test.robot_reboot.util import get_robot_reboot_game

POSITIONING_ENCODER_NAME = 'maze-and-robot-positioning-encoder'


class TestMazeAndRobotPositioningEncoder(unittest.TestCase):
    def test_name(self):
        encoder = MazeAndRobotPositioningEncoder(get_robot_reboot_game())
        self.assertEqual(POSITIONING_ENCODER_NAME, encoder.name())

    def test_shape(self):
        encoder = MazeAndRobotPositioningEncoder(get_robot_reboot_game())
        self.assertEqual((31, 31, 13), encoder.shape())

    # def test_encode(self):
    #     np.random.seed(26)
    #     encoder = MazeAndRobotPositioningEncoder(get_robot_reboot_game())
    #     game, game_state, quadrants_ids = RobotRebootFactory().create(31, locate_robot_close_goal=True, max_movements=4)
    #     # Robot 2 needs to get home.
    #     matrix = encoder.encode(game_state)
    #     np.testing.assert_equal(matrix[:, :, 0], game.maze)
    #     self.__assert_robot(game_state.robots_positions, matrix, 0)
    #     self.__assert_empty_houses(0, matrix)
    #
    #     self.__assert_robot(game_state.robots_positions, matrix, 1)
    #     self.__assert_empty_houses(1, matrix)
    #
    #     self.__assert_robot(game_state.robots_positions, matrix, 2)
    #     self.assert_house(game.goal_house, matrix)
    #
    #     self.__assert_robot(game_state.robots_positions, matrix, 3)
    #     self.__assert_empty_houses(3, matrix)

    def test_encode_action(self):
        n_robots = 4
        encoder = MazeAndRobotPositioningEncoder(get_robot_reboot_game(n_robots=n_robots))
        actions = [RobotRebootAction(r, d) for r in range(n_robots) for d in Direction]
        for i in range(len(actions)):
            action = actions[i]
            self.assertEqual(i, encoder.encode_action(action))

    def test_decode_action(self):
        n_robots = 4
        encoder = MazeAndRobotPositioningEncoder(get_robot_reboot_game(n_robots=n_robots))
        actions = [RobotRebootAction(r, d) for r in range(n_robots) for d in Direction]
        for i in range(len(actions)):
            action = actions[i]
            self.assertEqual(action, encoder.decode_action_index(i))

    def __assert_robot(self, robots_positions, matrix, robot_id):
        x, y = robots_positions[robot_id]
        robot_pos = np.argwhere(matrix[:, :, robot_id * 2 + 1] == RobotRebootState.ROBOT_IN_CELL)
        self.assertEqual(robot_pos.shape, (1, 2), f'There is more than one robot in the layer for robot {robot_id}')
        self.assertEqual(robot_pos[0, 0], x, f'Robot {robot_id} is not in the right x position')
        self.assertEqual(robot_pos[0, 1], y, f'Robot {robot_id} is not in the right y position')

    def __assert_empty_houses(self, robot_id, matrix):
        rows, cols, layers = matrix.shape
        np.testing.assert_equal(np.zeros((rows, cols)), matrix[:, :, (robot_id + 1) * 2],
                                f'Robot {robot_id} house  should be empty')

    def assert_house(self, house, matrix):
        x, y = house.house
        self.assertEqual(matrix[x, y, (house.robot_id + 1) * 2], RobotRebootState.ROBOT_IN_CELL,
                         f"House for robot {house.robot_id} not in the correct layer")
