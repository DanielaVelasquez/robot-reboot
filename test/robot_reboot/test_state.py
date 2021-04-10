import unittest
import numpy as np
from unittest.mock import Mock, PropertyMock

from exceptions.alphazero.state import InvalidStateSequence
from src.robot_reboot.factory import RobotRebootFactory
from src.robot_reboot.game import RobotRebootGame
from src.robot_reboot.goal_house import RobotRebootGoalHouse
from src.robot_reboot.state import RobotRebootState
from exceptions.robot_reboot.state import EmptyRobotsPositionException, InvalidRobotsPositionException, \
    RobotsPositionOutOfMazeBoundsException, NumberRobotsNotMatchingException, InvalidRobotsList


def get_game(size=3, n_robots=2):
    house = RobotRebootGoalHouse(0, (0, 0))
    maze = np.array([[0 for j in range(size)] for i in range(size)])
    game = RobotRebootGame(n_robots, maze, house)
    return game


class TestRobotRebootState(unittest.TestCase):

    def test_init_with_sequence_i_sets_all_given_values(self):
        s = RobotRebootState(get_game(), [(0, 2), (0, 0)], sequence_i=3)
        self.assertEqual(s.sequence_i, 3)
        self.assertEqual(s.robots_positions, [(0, 2), (0, 0)])
        self.assertIsNotNone(s.game)

    def test_init_without_sequence_i_sets_sequence_i_to_zero(self):
        s = RobotRebootState(get_game(), [(0, 2), (0, 0)])
        self.assertEqual(s.sequence_i, 0)

    def test_init_fails_when_empty_list_robot_positions(self):
        self.assertRaises(EmptyRobotsPositionException, lambda: RobotRebootState(get_game(), []))

    def test_init_fails_when_sequence_i_below_zero(self):
        self.assertRaises(InvalidStateSequence, lambda: RobotRebootState(get_game(), [(0, 2), (0, 0)], sequence_i=-1))

    def test_str(self):
        s = RobotRebootState(get_game(), [(0, 2), (0, 0)])
        self.assertEqual(str(s), '[(0, 2), (0, 0)]')

    def test_is_robot__on_returns_true_when_robot_on_position(self):
        s = RobotRebootState(get_game(size=7), [(0, 2), (4, 6)])
        self.assertTrue(s.is_robot_on((4, 6)))

    def test_is_robot__on_returns_false_when_no_robot_on_position(self):
        s = RobotRebootState(get_game(size=7), [(0, 2), (4, 6)])
        self.assertFalse(s.is_robot_on((5, 6)))

    def test_init_fails_when_one_robot_position_with_negative_value(self):
        self.assertRaises(InvalidRobotsPositionException,
                          lambda: RobotRebootState(get_game(), [(0, -1), (0, 0)]))

    def test_init_fails_when_one_robot_x_position_out_of_maze_bounds(self):
        self.assertRaises(RobotsPositionOutOfMazeBoundsException,
                          lambda: RobotRebootState(get_game(), [(0, 2), (6, 0)]))

    def test_init_fails_when_one_robot_out_of_maze_bounds(self):
        self.assertRaises(RobotsPositionOutOfMazeBoundsException,
                          lambda: RobotRebootState(get_game(), [(0, 2), (6, 6)]))

    def test_init_fails_when_number_robots_in_state_not_match_robots_in_game(self):
        self.assertRaises(NumberRobotsNotMatchingException,
                          lambda: RobotRebootState(RobotRebootState(get_game(n_robots=1), [(1, 1), (0, 1)])))

    def test_init_fails_robots_not_list(self):
        self.assertRaises(InvalidRobotsList,
                          lambda: RobotRebootState(RobotRebootState(get_game(n_robots=1), {(1, 0)})))

    def test_get_state(self):
        np.random.seed(26)
        factory = RobotRebootFactory()
        game, state, _ = factory.create(11)
        s = state.get_matrix()
        self.assertEqual((11, 11, 5), s.shape, "State should have the size of the maze for rows and columns and"
                                               "one layer for the maze"
                                               "one layer for robot 0 position"
                                               "one layer for robot 0 house goal, "
                                               "one layer for robot 1 position and "
                                               "one layer for robot 1 house goal")
        np.testing.assert_equal(game.maze, s[:, :, 0], "First layer should be the maze")
        np.testing.assert_equal(np.array(([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])), s[:, :, 1], "Second layer should have a one where the first robot is located and the rest should be zero")

        np.testing.assert_equal(np.array(([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])), s[:, :, 2],
            "Third layer should have a one where the first robot's goal is located and the rest should be zero")

        np.testing.assert_equal(np.array(([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])), s[:, :, 3],
            "Fourth layer should have a one where the second robot is located and the rest should be zero")

        np.testing.assert_equal(np.zeros((11, 11)), s[:, :, 4],
                                "Fifth layer should be all zero because second robot doesn't need to get to its house")

    def test_get_matrix(self):
        np.random.seed(26)
        game, state, quadrants_ids = RobotRebootFactory().create(31, locate_robot_close_goal=True, max_movements=4)
        # Knowledge robot 2 needs to get home with seed 26
        matrix = state.get_matrix()
        rows, cols, layers = matrix.shape
        np.testing.assert_equal(matrix[:, :, 0], game.maze)
        self.assert_robot(state.robots_positions, matrix, 0)
        self.assert_empty_houses(0, matrix)
        self.assert_robot(state.robots_positions, matrix, 1)
        self.assert_empty_houses(1, matrix)
        self.assert_robot(state.robots_positions, matrix, 2)
        self.assert_house(game.goal_house, matrix)
        self.assert_robot(state.robots_positions, matrix, 3)
        self.assert_empty_houses(3, matrix)

        game.goal_house.robot_id

    def assert_robot(self, robots_positions, matrix, robot_id):
        x, y = robots_positions[robot_id]
        self.assertEqual(matrix[x, y, robot_id * 2 + 1], RobotRebootState.ROBOT_IN_CELL,
                         f'Robot {robot_id} not in the correct position')

    def assert_empty_houses(self, robot_id, matrix):
        rows, cols, layers = matrix.shape
        np.testing.assert_equal(np.zeros((rows, cols)), matrix[:, :, (robot_id + 1) * 2],
                                f'Robot {robot_id} house is not empty')

    def assert_house(self, house, matrix):
        x, y = house.house
        self.assertEqual(matrix[x, y, (house.robot_id + 1) * 2], RobotRebootState.ROBOT_IN_CELL, f"House for robot {house.robot_id} not in the correct layer")
