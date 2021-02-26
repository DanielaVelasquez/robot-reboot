import unittest

import numpy as np

from src.robot_reboot.game import RobotRebootGame, Direction, RobotRebootGoalHouse, RobotRebootState


class TestGame(unittest.TestCase):
    def test_init(self):
        house = RobotRebootGoalHouse(1, (0, 0))
        maze = np.array([[0, 2, 0], [4, 5, 6], [0, 8, 0]])
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
        house = RobotRebootGoalHouse(1, (0, 0))
        maze = np.array([[0, 2, 0], [4, 5, 6], [0, 8, 0]])
        self.assertRaises(AssertionError, lambda: RobotRebootGame(0, maze, house))

    def test_init_fails_when_n_robots_is_below_zero(self):
        house = RobotRebootGoalHouse(1, (0, 0))
        maze = np.array([[0, 2, 0], [4, 5, 6], [0, 8, 0]])
        self.assertRaises(AssertionError, lambda: RobotRebootGame(-1, maze, house))

    def test_init_fails_when_maze_has_walls_on_robot_cells(self):
        house = RobotRebootGoalHouse(1, (0, 0))
        maze = np.array([[1, 2, 0], [4, 5, 6], [0, 8, 0]])
        self.assertRaises(AssertionError, lambda: RobotRebootGame(0, maze, house))

    def test_init_fails_when_house_row_out_maze(self):
        house = RobotRebootGoalHouse(1, (4, 0))
        maze = np.array([[0, 2, 0], [4, 5, 6], [0, 8, 0]])
        self.assertRaises(AssertionError, lambda: RobotRebootGame(0, maze, house))

    def test_init_fails_when_house_column_out_maze(self):
        house = RobotRebootGoalHouse(1, (0, 4))
        maze = np.array([[0, 2, 0], [4, 5, 6], [0, 8, 0]])
        self.assertRaises(AssertionError, lambda: RobotRebootGame(0, maze, house))

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
