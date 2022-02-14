import unittest

import numpy as np

from src.encoders.maze_and_robot_positioning_encoder import MazeAndRobotPositioningEncoder
from src.robot_reboot.action import RobotRebootAction
from src.robot_reboot.direction import Direction
from src.robot_reboot.game import RobotRebootGame
from src.robot_reboot.goal_house import RobotRebootGoalHouse
from src.robot_reboot.state import RobotRebootState
from test.robot_reboot.util import get_robot_reboot_game

POSITIONING_ENCODER_NAME = 'maze-and-robot-positioning-encoder'


def get_matrix_with_1s_in_positions(cols, rows, positions):
    robot_position_layer = np.zeros((rows, cols))
    for x, y in positions:
        robot_position_layer[x, y] = 1
    return robot_position_layer


class TestMazeAndRobotPositioningEncoder(unittest.TestCase):
    def test_name(self):
        encoder = MazeAndRobotPositioningEncoder(get_robot_reboot_game())
        self.assertEqual(POSITIONING_ENCODER_NAME, encoder.name())

    def test_shape(self):
        encoder = MazeAndRobotPositioningEncoder(get_robot_reboot_game())
        self.assertEqual((31, 31, 13), encoder.shape())

    def test_encode(self):
        goal_house_pos = (0, 0)
        house = RobotRebootGoalHouse(1, goal_house_pos)
        maze = np.array([[0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0]
                         ])
        game = RobotRebootGame(4, maze, house)
        robot_1 = (2, 2)
        robot_2 = (4, 0)
        robot_3 = (4, 2)
        robot_4 = (4, 4)
        game_state = RobotRebootState(game, [robot_1, robot_2, robot_3, robot_4])
        encoder = MazeAndRobotPositioningEncoder(game)
        encoded_state = encoder.encode(game_state)
        self.assertEqual((5, 5, 13), encoded_state.shape)
        # First layer is the maze
        np.testing.assert_equal(encoded_state[:, :, 0], maze)
        #Robots layers
        self.__assert_robot_layers(encoded_state[:, :, 1:4], robot_1, [(0, 2), (2, 0), (2, 4)])
        self.__assert_robot_layers(encoded_state[:, :, 4:7], robot_2, [(0, 0)], goal_house_pos)
        self.__assert_robot_layers(encoded_state[:, :, 7:10], robot_3, [])
        self.__assert_robot_layers(encoded_state[:, :, 10:], robot_4, [(0, 4)])

    def __assert_robot_layers(self, robot_encoded_layers, robot_position, robot_future_positions_list,
                              robot_house_position=None):
        rows, cols, layers = robot_encoded_layers.shape
        x, y = robot_position
        # Robot position layer
        robot_position_layer = get_matrix_with_1s_in_positions(cols, rows, [(x, y)])
        np.testing.assert_equal(robot_position_layer, robot_encoded_layers[:, :, 0])
        # Robot house layer
        if robot_house_position:
            robot_house_layer = get_matrix_with_1s_in_positions(cols, rows, [robot_house_position])
        else:
            robot_house_layer = np.zeros((rows, cols))
        np.testing.assert_equal(robot_house_layer, robot_encoded_layers[:, :, 1])

        robot_future_positions_layer = get_matrix_with_1s_in_positions(cols, rows, robot_future_positions_list)
        np.testing.assert_equal(robot_future_positions_layer, robot_encoded_layers[:, :, 2])

        # Robot future moves layer

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
