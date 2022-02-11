import unittest

import numpy as np

from src.exceptions.game.state import InvalidStateSequence
from src.exceptions.robot_reboot.state import EmptyRobotsPositionException, InvalidRobotsPositionException, \
    RobotsPositionOutOfMazeBoundsException, NumberRobotsNotMatchingException, InvalidRobotsList, \
    RobotsPositionsOnWallsPositionsExceptions
from src.robot_reboot.game import RobotRebootGame
from src.robot_reboot.goal_house import RobotRebootGoalHouse
from src.robot_reboot.state import RobotRebootState


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
                          lambda: RobotRebootState(RobotRebootState(get_game(n_robots=1), [(0, 2), (2, 2)])))

    def test_init_fails_robots_not_list(self):
        self.assertRaises(InvalidRobotsList,
                          lambda: RobotRebootState(RobotRebootState(get_game(n_robots=1), {(1, 1)})))

    def test_init_fails_robots_on_wall_position(self):
        self.assertRaises(RobotsPositionsOnWallsPositionsExceptions,
                          lambda: RobotRebootState(RobotRebootState(get_game(n_robots=1), [(1, 0)])))

    def test_robots_count(self):
        s = RobotRebootState(get_game(), [(0, 2), (0, 0)])
        self.assertEqual(2, s.robots_count)

    def assert_robot(self, robots_positions, matrix, robot_id):
        x, y = robots_positions[robot_id]
        robot_pos = np.argwhere(matrix[:, :, robot_id * 2 + 1] == RobotRebootState.ROBOT_IN_CELL)
        self.assertEqual(robot_pos.shape, (1, 2), f'There is more than one robot in the layer for robot {robot_id}')
        self.assertEqual(robot_pos[0, 0], x, f'Robot {robot_id} is not in the right x position')
        self.assertEqual(robot_pos[0, 1], y, f'Robot {robot_id} is not in the right y position')

    def assert_empty_houses(self, robot_id, matrix):
        rows, cols, layers = matrix.shape
        np.testing.assert_equal(np.zeros((rows, cols)), matrix[:, :, (robot_id + 1) * 2],
                                f'Robot {robot_id} house  should be empty')

    def assert_house(self, house, matrix):
        x, y = house.house
        self.assertEqual(matrix[x, y, (house.robot_id + 1) * 2], RobotRebootState.ROBOT_IN_CELL,
                         f"House for robot {house.robot_id} not in the correct layer")
