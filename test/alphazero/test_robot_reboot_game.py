import unittest

import numpy as np

from src.alphazero.robot_reboot_game import RobotRebootGame, RobotRebootAction, Maze
from src.alphazero.util import get_movement_same_direction_for_wall, get_cell_at, get_wall_at_direction, \
    get_opposite_direction, Direction
from src.models.robotreboot import Goal


class TestRobotRebootGame(unittest.TestCase):

    def test_can_move_when_no_walls(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        robot = (1, 1)
        for move in RobotRebootGame.MOVEMENTS:
            game = RobotRebootGame(maze, [robot], Goal(0, (0, 0)))
            self.assertTrue(game.can_move(RobotRebootAction(0, move)),
                            f'Movement {move} should be allowed, there are no walls in the maze')

    def test_can_move_when_wall_on_same_position_as_robot_on_the_way(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        robot = (1, 1)
        for wall in Maze.WALLS:
            maze.cells[1, 1] = wall
            game = RobotRebootGame(maze, [robot], Goal(0, (0, 0)))
            movement = get_movement_same_direction_for_wall(wall)
            self.assertFalse(game.can_move(RobotRebootAction(0, movement)),
                             f'Movement {movement} should NOT be allowed, there is a wall in the way' + str(maze.cells))

    def test_can_move_when_wall_on_next_cell_of_the_movement(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        robot = (1, 1)
        for movement_direction in RobotRebootGame.MOVEMENTS:
            cell = get_cell_at(movement_direction, robot, 3, 3)
            wall = get_wall_at_direction(get_opposite_direction(movement_direction))
            maze.cells[cell] = wall
            game = RobotRebootGame(maze, [robot], Goal(0, (0, 0)))
            self.assertFalse(game.can_move(RobotRebootAction(0, movement_direction)),
                             f'Movement {movement_direction} should NOT be allowed, there is a wall in the way')

    def test_execute_move_and_execute_undo_move_on_north_direction(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        robot_initial_position = (1, 1)
        game = RobotRebootGame(maze, [robot_initial_position], Goal(0, (0, 0)))
        action = RobotRebootAction(0, Direction.NORTH)
        game.execute_move(action)
        self.assertEqual([(0, 1)], game.robots)
        game.execute_undo_move(action)
        self.assertEqual([robot_initial_position], game.robots)

    def test_execute_move_and_execute_undo_move_on_south_direction(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        robot_initial_position = (1, 1)
        game = RobotRebootGame(maze, [robot_initial_position], Goal(0, (0, 0)))
        action = RobotRebootAction(0, Direction.SOUTH)
        game.execute_move(action)
        self.assertEqual([(2, 1)], game.robots)
        game.execute_undo_move(action)
        self.assertEqual([robot_initial_position], game.robots)

    def test_execute_move_and_execute_undo_move_on_west_direction(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        robot_initial_position = (1, 1)
        game = RobotRebootGame(maze, [robot_initial_position], Goal(0, (0, 0)))
        action = RobotRebootAction(0, Direction.WEST)
        game.execute_move(action)
        self.assertEqual([(1, 0)], game.robots)
        game.execute_undo_move(action)
        self.assertEqual([robot_initial_position], game.robots)

    def test_execute_move_and_execute_undo_move_on_east_direction(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        robot_initial_position = (1, 1)
        game = RobotRebootGame(maze, [robot_initial_position], Goal(0, (0, 0)))
        action = RobotRebootAction(0, Direction.EAST)
        game.execute_move(action)
        self.assertEqual([(1, 2)], game.robots)
        game.execute_undo_move(action)
        self.assertEqual([robot_initial_position], game.robots)

    def test_execute_undo_move_after_three_movements(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        robot_initial_position = (1, 1)
        game = RobotRebootGame(maze, [robot_initial_position], Goal(0, (0, 0)))
        game.execute_move(RobotRebootAction(0, Direction.NORTH))
        game.execute_move(RobotRebootAction(0, Direction.WEST))
        game.execute_move(RobotRebootAction(0, Direction.SOUTH))
        expected_positions = game.robots.copy()
        action = RobotRebootAction(0, Direction.EAST)
        game.execute_move(action)
        game.execute_undo_move(action)
        self.assertEqual(expected_positions, game.robots)

    def test_get_all_actions_with_two_robots(self):
        maze = Maze(np.full((3, 6), Maze.EMPTY))
        robots = [(1, 1), (1, 4)]
        game = RobotRebootGame(maze, robots, Goal(0, (0, 0)))
        actions = game.get_all_actions()
        expected_actions = [
            RobotRebootAction(0, Direction.NORTH),
            RobotRebootAction(0, Direction.SOUTH),
            RobotRebootAction(0, Direction.EAST),
            RobotRebootAction(0, Direction.WEST),
            RobotRebootAction(1, Direction.NORTH),
            RobotRebootAction(1, Direction.SOUTH),
            RobotRebootAction(1, Direction.EAST),
            RobotRebootAction(1, Direction.WEST)
        ]
        self.assertCountEqual(expected_actions, actions)

    def test_get_valid_actions_when_only_north_allowed(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EAST_WALL, Maze.EMPTY, Maze.WEST_WALL],
            [Maze.EMPTY, Maze.NORTH_WALL, Maze.EMPTY],
        ]))
        robots = [(1, 1)]
        expected_actions = [
            RobotRebootAction(0, Direction.NORTH),

        ]
        game = RobotRebootGame(maze, robots, Goal(0, (0, 0)))
        valid_actions = game.get_valid_actions()
        self.assertCountEqual(expected_actions, valid_actions)

    def test_get_valid_actions_when_only_south_allowed(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.SOUTH_WALL, Maze.EMPTY],
            [Maze.EAST_WALL, Maze.EMPTY, Maze.WEST_WALL],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
        ]))
        robots = [(1, 1)]
        expected_actions = [
            RobotRebootAction(0, Direction.SOUTH),

        ]
        game = RobotRebootGame(maze, robots, Goal(0, (0, 0)))
        valid_actions = game.get_valid_actions()
        self.assertCountEqual(expected_actions, valid_actions)

    def test_get_valid_actions_when_only_west_allowed(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.SOUTH_WALL, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.WEST_WALL],
            [Maze.EMPTY, Maze.NORTH_WALL, Maze.EMPTY],
        ]))
        robots = [(1, 1)]
        expected_actions = [
            RobotRebootAction(0, Direction.WEST),

        ]
        game = RobotRebootGame(maze, robots, Goal(0, (0, 0)))
        valid_actions = game.get_valid_actions()
        self.assertCountEqual(expected_actions, valid_actions)

    def test_get_valid_actions_when_only_east_allowed(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.SOUTH_WALL, Maze.EMPTY],
            [Maze.EAST_WALL, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.NORTH_WALL, Maze.EMPTY],
        ]))
        robots = [(1, 1)]
        expected_actions = [
            RobotRebootAction(0, Direction.EAST),

        ]
        game = RobotRebootGame(maze, robots, Goal(0, (0, 0)))
        valid_actions = game.get_valid_actions()
        self.assertCountEqual(expected_actions, valid_actions)

    def test_get_valid_actions_with_two_robots(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.SOUTH_WALL, Maze.EMPTY],
            [Maze.EAST_WALL, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.NORTH_WALL, Maze.EMPTY],
        ]))
        robots = [(1, 1), (0, 2)]
        expected_actions = [
            RobotRebootAction(0, Direction.EAST),
            RobotRebootAction(1, Direction.WEST),
            RobotRebootAction(1, Direction.SOUTH),
        ]
        game = RobotRebootGame(maze, robots, Goal(0, (0, 0)))
        valid_actions = game.get_valid_actions()
        self.assertCountEqual(expected_actions, valid_actions)

    def test_move_north(self):
        maze = Maze(np.array([
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.EMPTY],
        ]))
        robots = [(4, 0)]
        game = RobotRebootGame(maze, robots, Goal(0, (0, 0)))
        game.execute_move(RobotRebootAction(0, Direction.NORTH))
        self.assertEqual([(0,0)], game.robots)
