import unittest

import numpy as np

from src.robot_reboot.exceptions import NoRobotsGameException, InvalidMazeException, RobotHouseOutOfMazeBoundsException, \
    MazeNotSquareException, MazeSizeInvalidException, RobotHouseInvalidRobotIdException
from src.robot_reboot.game import RobotRebootGame, Direction, RobotRebootGoalHouse, RobotRebootState, RobotRebootAction


class TestGame(unittest.TestCase):
    def test_init(self):
        house = RobotRebootGoalHouse(1, (0, 0))
        maze = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        g = RobotRebootGame(2, maze, house)
        self.assertEqual(g.n_robots, 2)
        self.assertEqual(g.goal_house, house)
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
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertRaises(NoRobotsGameException, lambda: RobotRebootGame(0, maze, house))

    def test_init_fails_when_n_robots_is_below_zero(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertRaises(NoRobotsGameException, lambda: RobotRebootGame(-1, maze, house))

    def test_init_fails_when_maze_not_only_contains_zeros_and_ones(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[1, 2, 0], [4, 5, 6], [0, 8, 0]])
        self.assertRaises(InvalidMazeException, lambda: RobotRebootGame(1, maze, house))

    def test_init_fails_when_maze_has_walls_on_robot_cells(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertRaises(InvalidMazeException, lambda: RobotRebootGame(1, maze, house))

    def test_init_fails_when_house_row_out_maze(self):
        house = RobotRebootGoalHouse(0, (4, 0))
        maze = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertRaises(RobotHouseOutOfMazeBoundsException, lambda: RobotRebootGame(1, maze, house))

    def test_init_fails_when_house_column_out_maze(self):
        house = RobotRebootGoalHouse(0, (0, 4))
        maze = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertRaises(RobotHouseOutOfMazeBoundsException, lambda: RobotRebootGame(1, maze, house))

    def test_init_fails_when_robot_house_invalid(self):
        house = RobotRebootGoalHouse(1, (0, 0))
        maze = np.array([[0, 0], [0, 0]])
        self.assertRaises(RobotHouseInvalidRobotIdException, lambda: RobotRebootGame(1, maze, house))

    def test_init_fails_when_maze_not_square(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0, 0], [0, 0, 0]])
        self.assertRaises(MazeNotSquareException, lambda: RobotRebootGame(1, maze, house))

    def test_init_fails_when_maze_size_not_odd(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0], [0, 0]])
        self.assertRaises(MazeSizeInvalidException, lambda: RobotRebootGame(1, maze, house))

    def test_get_value_when_robot_reached_its_house(self):
        """
                |  R1 |      |  R2 |
                |     |      |     |
                |     |      |     |
        """
        house_pos = (0, 0)
        house = RobotRebootGoalHouse(0, house_pos)
        maze = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        game = RobotRebootGame(2, maze, house)

        s = RobotRebootState(game, [house_pos, (0, 2)])
        self.assertEqual(1, game.get_value(s))

    def test_get_value_when_no_robot_reached_goal_house(self):
        """
                |     |      |  R2 |
                |     |      |     |
                |     |      |  R1 |
        """
        house_pos = (0, 0)
        house = RobotRebootGoalHouse(0, house_pos)
        maze = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        game = RobotRebootGame(2, maze, house)

        s = RobotRebootState(game, [(2, 2), (0, 2)])
        self.assertEqual(0, game.get_value(s))

    def test_get_value_when_wrong_robot_reached_goal_house(self):
        """
                |  R2 |      |  R1 |
                |     |      |     |
                |     |      |     |
        """
        house_pos = (0, 0)
        house = RobotRebootGoalHouse(0, house_pos)
        maze = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        game = RobotRebootGame(2, maze, house)

        s = RobotRebootState(game, [(0, 2), house_pos])
        self.assertEqual(0, game.get_value(s))

    def test_is_over_when_robot_reached_its_house(self):
        """
                |  R1 |      |  R2 |
                |     |      |     |
                |     |      |     |
        """
        house_pos = (0, 0)
        house = RobotRebootGoalHouse(0, house_pos)
        maze = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        game = RobotRebootGame(2, maze, house)

        s = RobotRebootState(game, [house_pos, (0, 2)])
        self.assertTrue(game.is_over(s))

    def test_is_over_when_no_robot_reached_goal_house(self):
        """
                |     |      |  R2 |
                |     |      |     |
                |     |      |  R1 |
        """
        house_pos = (0, 0)
        house = RobotRebootGoalHouse(0, house_pos)
        maze = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        game = RobotRebootGame(2, maze, house)

        s = RobotRebootState(game, [(2, 2), (0, 2)])
        self.assertFalse(game.is_over(s))

    def test_is_over_when_wrong_robot_reached_goal_house(self):
        """
                |  R2 |      |  R1 |
                |     |      |     |
                |     |      |     |
        """
        house_pos = (0, 0)
        house = RobotRebootGoalHouse(0, house_pos)
        maze = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        game = RobotRebootGame(2, maze, house)

        s = RobotRebootState(game, [(0, 2), house_pos])
        self.assertFalse(game.is_over(s))

    def test_get_score_for_root_state(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        game = RobotRebootGame(2, maze, house)

        s = RobotRebootState(game, [(0, 2), (2, 2)])
        self.assertEqual(0, game.get_score(s))

    def test_get_score_for_state_with_sequence_value(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        game = RobotRebootGame(2, maze, house)

        s = RobotRebootState(game, [(0, 2), (2, 2)], sequence_i=20)
        self.assertEqual(-20, game.get_score(s))

    def test_apply_raises_exception_when_unsupported_direction(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(2, 0), (0, 0)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.NORTH)
        self.assertRaises(Exception, game.apply(a, s))

    def test_apply_robot_moves_north_when_no_walls_in_maze(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0 for i in range(9)] for j in range(9)])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(4, 4), (0, 0)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.NORTH)
        next_state = game.apply(a, s)
        self.assertEqual([(0, 4), (0, 0)], next_state.robots_positions, "Robot should move to the north of the maze")
        self.assertEqual(1, next_state.sequence_i, "Sequence should be update")
        self.assertEqual(robots_positions, s.robots_positions, "Robots position on initial state should not be altered")
        self.assertEqual(0, s.sequence_i, "Sequence on initial state should not be altered")

    def test_apply_robot_moves_south_when_no_walls_in_maze(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0 for i in range(9)] for j in range(9)])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(4, 4), (0, 0)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.SOUTH)
        next_state = game.apply(a, s)
        self.assertEqual([(8, 4), (0, 0)], next_state.robots_positions, "Robot should move to the south of the maze")
        self.assertEqual(1, next_state.sequence_i, "Sequence should be update")
        self.assertEqual(robots_positions, s.robots_positions, "Robots position on initial state should not be altered")
        self.assertEqual(0, s.sequence_i, "Sequence on initial state should not be altered")

    def test_apply_robot_moves_west_when_no_walls_in_maze(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0 for i in range(9)] for j in range(9)])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(4, 4), (0, 0)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.WEST)
        next_state = game.apply(a, s)
        self.assertEqual([(4, 0), (0, 0)], next_state.robots_positions,
                         "Robot should move to the west side of the maze")
        self.assertEqual(1, next_state.sequence_i, "Sequence should be update")
        self.assertEqual(robots_positions, s.robots_positions, "Robots position on initial state should not be altered")
        self.assertEqual(0, s.sequence_i, "Sequence on initial state should not be altered")

    def test_apply_robot_moves_east_when_no_walls_in_maze(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0 for i in range(9)] for j in range(9)])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(4, 4), (0, 0)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.EAST)
        next_state = game.apply(a, s)
        self.assertEqual([(4, 8), (0, 0)], next_state.robots_positions,
                         "Robot should move to the east side of the maze")
        self.assertEqual(1, next_state.sequence_i, "Sequence should be update")
        self.assertEqual(robots_positions, s.robots_positions, "Robots position on initial state should not be altered")
        self.assertEqual(0, s.sequence_i, "Sequence on initial state should not be altered")

    def test_apply_robot_moves_north_when_walls_first_robots_next(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         #                        R
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                         ])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(8, 8), (0, 8)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.NORTH)
        next_state = game.apply(a, s)
        self.assertEqual([(4, 8), (0, 8)], next_state.robots_positions,
                         "Robot should move to the north of the maze, until it finds the wall")
        self.assertEqual(1, next_state.sequence_i, "Sequence should be update")
        self.assertEqual(robots_positions, s.robots_positions, "Robots position on initial state should not be altered")
        self.assertEqual(0, s.sequence_i, "Sequence on initial state should not be altered")

    def test_apply_robot_moves_south_when_walls_first_robots_next(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         #                        R
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                         ])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(8, 8), (16, 8)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.SOUTH)
        next_state = game.apply(a, s)
        self.assertEqual([(12, 8), (16, 8)], next_state.robots_positions,
                         "Robot should move to the south of the maze, until it finds the wall")
        self.assertEqual(1, next_state.sequence_i, "Sequence should be update")
        self.assertEqual(robots_positions, s.robots_positions, "Robots position on initial state should not be altered")
        self.assertEqual(0, s.sequence_i, "Sequence on initial state should not be altered")

    def test_apply_robot_moves_west_when_walls_first_robots_next(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         #                        R
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                         ])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(8, 8), (8, 0)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.WEST)
        next_state = game.apply(a, s)
        self.assertEqual([(8, 4), (8, 0)], next_state.robots_positions,
                         "Robot should move to the west of the maze, until it finds the wall")
        self.assertEqual(1, next_state.sequence_i, "Sequence should be update")
        self.assertEqual(robots_positions, s.robots_positions, "Robots position on initial state should not be altered")
        self.assertEqual(0, s.sequence_i, "Sequence on initial state should not be altered")

    def test_apply_robot_moves_east_when_walls_first_robot_next(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                         #                        R
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                         ])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(8, 8), (8, 16)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.EAST)
        next_state = game.apply(a, s)
        self.assertEqual([(8, 12), (8, 16)], next_state.robots_positions,
                         "Robot should move to the east of the maze, until it finds the wall")
        self.assertEqual(1, next_state.sequence_i, "Sequence should be update")
        self.assertEqual(robots_positions, s.robots_positions, "Robots position on initial state should not be altered")
        self.assertEqual(0, s.sequence_i, "Sequence on initial state should not be altered")

    def test_apply_robot_not_moves_north_when_wall(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]
                         ])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(2, 2), (0, 0)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.NORTH)
        next_state = game.apply(a, s)
        self.assertEqual(robots_positions, next_state.robots_positions,
                         "Robot shouldn't move to the north of the maze there is a wall in the next cell")
        self.assertEqual(1, next_state.sequence_i, "Sequence should be update")
        self.assertEqual(robots_positions, s.robots_positions, "Robots position on initial state should not be altered")
        self.assertEqual(0, s.sequence_i, "Sequence on initial state should not be altered")

    def test_apply_robot_not_moves_south_when_wall(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0]
                         ])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(2, 2), (0, 0)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.SOUTH)
        next_state = game.apply(a, s)
        self.assertEqual(robots_positions, next_state.robots_positions,
                         "Robot shouldn't move to the south of the maze there is a wall in the next cell")
        self.assertEqual(1, next_state.sequence_i, "Sequence should be update")
        self.assertEqual(robots_positions, s.robots_positions, "Robots position on initial state should not be altered")
        self.assertEqual(0, s.sequence_i, "Sequence on initial state should not be altered")

    def test_apply_robot_not_moves_west_when_wall(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]
                         ])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(2, 2), (0, 0)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.WEST)
        next_state = game.apply(a, s)
        self.assertEqual(robots_positions, next_state.robots_positions,
                         "Robot shouldn't move to the west of the maze there is a wall in the next cell")
        self.assertEqual(1, next_state.sequence_i, "Sequence should be update")
        self.assertEqual(robots_positions, s.robots_positions, "Robots position on initial state should not be altered")
        self.assertEqual(0, s.sequence_i, "Sequence on initial state should not be altered")

    def test_apply_robot_not_moves_east_when_wall(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]
                         ])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(2, 2), (0, 0)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.EAST)
        next_state = game.apply(a, s)
        self.assertEqual(robots_positions, next_state.robots_positions,
                         "Robot shouldn't move to the east of the maze there is a wall in the next cell")
        self.assertEqual(1, next_state.sequence_i, "Sequence should be update")
        self.assertEqual(robots_positions, s.robots_positions, "Robots position on initial state should not be altered")
        self.assertEqual(0, s.sequence_i, "Sequence on initial state should not be altered")

    def test_apply_robot_moves_north_when_robots_first_walls_next(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         #                        R
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                         ])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(8, 8), (4, 8)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.NORTH)
        next_state = game.apply(a, s)
        self.assertEqual([(6, 8), (4, 8)], next_state.robots_positions,
                         f"Robot should move to the north of the maze, until it finds the robot on {robots_positions[1]}")
        self.assertEqual(1, next_state.sequence_i, "Sequence should be update")
        self.assertEqual(robots_positions, s.robots_positions, "Robots position on initial state should not be altered")
        self.assertEqual(0, s.sequence_i, "Sequence on initial state should not be altered")

    def test_apply_robot_moves_south_when_robots_first_walls_next(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         #                        R
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                         ])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(8, 8), (12, 8)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.SOUTH)
        next_state = game.apply(a, s)
        self.assertEqual([(10, 8), (12, 8)], next_state.robots_positions,
                         f"Robot should move to the south of the maze, until it finds the robot on {robots_positions[1]}")
        self.assertEqual(1, next_state.sequence_i, "Sequence should be update")
        self.assertEqual(robots_positions, s.robots_positions, "Robots position on initial state should not be altered")
        self.assertEqual(0, s.sequence_i, "Sequence on initial state should not be altered")

    def test_apply_robot_moves_west_when_robots_first_walls_next(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         #                        R
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                         ])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(8, 8), (8, 4)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.WEST)
        next_state = game.apply(a, s)
        self.assertEqual([(8, 6), (8, 4)], next_state.robots_positions,
                         f"Robot should move to the west of the maze, until it finds the robot on {robots_positions[1]}")
        self.assertEqual(1, next_state.sequence_i, "Sequence should be update")
        self.assertEqual(robots_positions, s.robots_positions, "Robots position on initial state should not be altered")
        self.assertEqual(0, s.sequence_i, "Sequence on initial state should not be altered")

    def test_apply_robot_moves_east_when_robots_first_walls_next(self):
        house = RobotRebootGoalHouse(0, (0, 0))
        maze = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                         #                        R
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                         ])
        game = RobotRebootGame(2, maze, house)

        robots_positions = [(8, 8), (8, 12)]
        s = RobotRebootState(game, robots_positions)
        a = RobotRebootAction(0, Direction.EAST)
        next_state = game.apply(a, s)
        self.assertEqual([(8, 10), (8, 12)], next_state.robots_positions,
                         f"Robot should move to the east of the maze, until it finds the robot on {robots_positions[1]}")
        self.assertEqual(1, next_state.sequence_i, "Sequence should be update")
        self.assertEqual(robots_positions, s.robots_positions, "Robots position on initial state should not be altered")
        self.assertEqual(0, s.sequence_i, "Sequence on initial state should not be altered")
