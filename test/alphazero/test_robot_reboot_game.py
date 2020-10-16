import unittest

import numpy as np

from src.alphazero.robot_reboot_game import RobotRebootGame, RobotRebootAction, Maze, RobotRebootGoal
from src.alphazero.util import get_movement_same_direction_for_wall, get_cell_at, get_wall_at_direction, \
    get_opposite_direction, Direction, get_not_blocking_walls_when_robot_on_cell_moving_to


class TestRobotRebootGame(unittest.TestCase):

    def test_can_move_when_no_walls(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        robot = (1, 1)
        for move in RobotRebootGame.MOVEMENTS:
            game = RobotRebootGame(maze, [robot], RobotRebootGoal(0, (0, 0)))
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
            game = RobotRebootGame(maze, [robot], RobotRebootGoal(0, (0, 0)))
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
            game = RobotRebootGame(maze, [robot], RobotRebootGoal(0, (0, 0)))
            self.assertFalse(game.can_move(RobotRebootAction(0, movement_direction)),
                             f'Movement {movement_direction} should NOT be allowed, there is a wall in the way')

    def test_move_when_robot_on_next_cell_of_the_movement(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        robot = (1, 1)
        robots = [robot]
        for movement_direction in RobotRebootGame.MOVEMENTS:
            cell = get_cell_at(movement_direction, robot, 3, 3)
            robots.append(cell)
            game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
            game.move(RobotRebootAction(0, movement_direction))
            self.assertEqual(robots, game.robots, 'Robots position should not change, there is a robot on its way')

    def test_move_and_undo_move_on_north_direction(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        robot_initial_position = (1, 1)
        game = RobotRebootGame(maze, [robot_initial_position], RobotRebootGoal(0, (0, 0)))
        action = RobotRebootAction(0, Direction.NORTH)
        game.move(action)
        self.assertEqual([(0, 1)], game.robots)
        game.undo_move()
        self.assertEqual([robot_initial_position], game.robots)

    def test_move_and_undo_move_on_south_direction(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        robot_initial_position = (1, 1)
        game = RobotRebootGame(maze, [robot_initial_position], RobotRebootGoal(0, (0, 0)))
        action = RobotRebootAction(0, Direction.SOUTH)
        game.move(action)
        self.assertEqual([(2, 1)], game.robots)
        game.undo_move()
        self.assertEqual([robot_initial_position], game.robots)

    def test_move_and_undo_move_on_west_direction(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        robot_initial_position = (1, 1)
        game = RobotRebootGame(maze, [robot_initial_position], RobotRebootGoal(0, (0, 0)))
        action = RobotRebootAction(0, Direction.WEST)
        game.move(action)
        self.assertEqual([(1, 0)], game.robots)
        game.undo_move()
        self.assertEqual([robot_initial_position], game.robots)

    def test_move_and_undo_move_on_east_direction(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        robot_initial_position = (1, 1)
        game = RobotRebootGame(maze, [robot_initial_position], RobotRebootGoal(0, (0, 0)))
        action = RobotRebootAction(0, Direction.EAST)
        game.move(action)
        self.assertEqual([(1, 2)], game.robots)
        game.undo_move()
        self.assertEqual([robot_initial_position], game.robots)

    def test_undo_move_after_three_movements(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        robot_initial_position = (1, 1)
        game = RobotRebootGame(maze, [robot_initial_position], RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.NORTH))
        game.move(RobotRebootAction(0, Direction.WEST))
        game.move(RobotRebootAction(0, Direction.SOUTH))
        expected_positions = game.robots.copy()
        action = RobotRebootAction(0, Direction.EAST)
        game.move(action)
        game.undo_move()
        self.assertEqual(expected_positions, game.robots)

    def test_get_all_actions_with_two_robots(self):
        maze = Maze(np.full((3, 6), Maze.EMPTY))
        robots = [(1, 1), (1, 4)]
        game = RobotRebootGame(maze, robots, RobotRebootGoal(0, (0, 0)))
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
        game = RobotRebootGame(maze, robots, RobotRebootGoal(0, (0, 0)))
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
        game = RobotRebootGame(maze, robots, RobotRebootGoal(0, (0, 0)))
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
        game = RobotRebootGame(maze, robots, RobotRebootGoal(0, (0, 0)))
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
        game = RobotRebootGame(maze, robots, RobotRebootGoal(0, (0, 0)))
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
        game = RobotRebootGame(maze, robots, RobotRebootGoal(0, (0, 0)))
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
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.NORTH))
        self.assertEqual([(0, 0)], game.robots)

    def test_move_north_when_north_wall_on_robot_pos(self):
        maze = Maze(np.array([
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.NORTH_WALL],
        ]))
        robots = [(4, 0)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.NORTH))
        self.assertEqual(robots, game.robots)

    def test_move_north_when_not_blocking_wall_on_robot_pos(self):
        maze = Maze(np.array([
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.EMPTY],
        ]))
        direction = Direction.NORTH
        not_blocking_walls = get_not_blocking_walls_when_robot_on_cell_moving_to(direction)
        robot_pos = (4, 0)
        for wall in not_blocking_walls:
            maze.cells[robot_pos] = wall
            robots = [robot_pos]
            game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
            game.move(RobotRebootAction(0, direction))
            self.assertEqual([(0, 0)], game.robots)

    def test_move_north_when_north_wall_on_the_way(self):
        maze = Maze(np.array([
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.NORTH_WALL],
            [Maze.EMPTY],
            [Maze.EMPTY],
        ]))
        robots = [(4, 0)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.NORTH))
        self.assertEqual([(2, 0)], game.robots)

    def test_move_north_when_south_wall_on_the_way(self):
        maze = Maze(np.array([
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.SOUTH_WALL],
            [Maze.EMPTY],
            [Maze.EMPTY],
        ]))
        robots = [(4, 0)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.NORTH))
        self.assertEqual([(3, 0)], game.robots)

    def test_move_south(self):
        maze = Maze(np.array([
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.EMPTY],
        ]))
        robots = [(0, 0)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.SOUTH))
        self.assertEqual([(4, 0)], game.robots)

    def test_move_south_when_south_wall_on_robot_pos(self):
        maze = Maze(np.array([
            [Maze.SOUTH_WALL],
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.EMPTY],
        ]))
        robots = [(0, 0)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.NORTH))
        self.assertEqual(robots, game.robots)

    def test_move_south_when_not_blocking_wall_on_robot_pos(self):
        maze = Maze(np.array([
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.EMPTY],
        ]))
        direction = Direction.SOUTH
        not_blocking_walls = get_not_blocking_walls_when_robot_on_cell_moving_to(direction)
        robot_pos = (0, 0)
        for wall in not_blocking_walls:
            maze.cells[robot_pos] = wall
            robots = [robot_pos]
            game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
            game.move(RobotRebootAction(0, direction))
            self.assertEqual([(4, 0)], game.robots)

    def test_move_south_when_north_wall_on_the_way(self):
        maze = Maze(np.array([
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.NORTH_WALL],
            [Maze.EMPTY],
            [Maze.EMPTY],
        ]))
        robots = [(0, 0)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.SOUTH))
        self.assertEqual([(1, 0)], game.robots)

    def test_move_south_when_south_wall_on_the_way(self):
        maze = Maze(np.array([
            [Maze.EMPTY],
            [Maze.EMPTY],
            [Maze.SOUTH_WALL],
            [Maze.EMPTY],
            [Maze.EMPTY],
        ]))
        robots = [(0, 0)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.SOUTH))
        self.assertEqual([(2, 0)], game.robots)

    def test_move_east(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
        ]))
        robots = [(1, 0)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.EAST))
        self.assertEqual([(1, 4)], game.robots)

    def test_move_east_when_east_wall_on_robot_pos(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EAST_WALL, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
        ]))
        robots = [(1, 0)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.EAST))
        self.assertEqual(robots, game.robots)

    def test_move_east_when_not_blocking_wall_on_robot_pos(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
        ]))
        direction = Direction.EAST
        not_blocking_walls = get_not_blocking_walls_when_robot_on_cell_moving_to(direction)
        robot_pos = (1, 0)
        for wall in not_blocking_walls:
            maze.cells[robot_pos] = wall
            robots = [robot_pos]
            game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
            game.move(RobotRebootAction(0, direction))
            self.assertEqual([(1, 4)], game.robots)

    def test_move_east_when_east_wall_on_the_way(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EAST_WALL, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
        ]))
        robots = [(1, 0)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.EAST))
        self.assertEqual([(1, 2)], game.robots)

    def test_move_east_when_west_wall_on_the_way(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.WEST_WALL, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
        ]))
        robots = [(1, 0)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.EAST))
        self.assertEqual([(1, 1)], game.robots)

    def test_move_west(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
        ]))
        robots = [(1, 4)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.WEST))
        self.assertEqual([(1, 0)], game.robots)

    def test_move_west_when_west_wall_on_robot_pos(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.WEST_WALL],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
        ]))
        robots = [(1, 4)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.EAST))
        self.assertEqual(robots, game.robots)

    def test_move_west_when_not_blocking_wall_on_robot_pos(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
        ]))
        direction = Direction.WEST
        not_blocking_walls = get_not_blocking_walls_when_robot_on_cell_moving_to(direction)
        robot_pos = (1, 4)
        for wall in not_blocking_walls:
            maze.cells[robot_pos] = wall
            robots = [robot_pos]
            game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
            game.move(RobotRebootAction(0, direction))
            self.assertEqual([(1, 0)], game.robots)

    def test_move_west_when_east_wall_on_the_way(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EAST_WALL, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
        ]))
        robots = [(1, 4)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.WEST))
        self.assertEqual([(1, 3)], game.robots)

    def test_move_west_when_west_wall_on_the_way(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.WEST_WALL, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
        ]))
        robots = [(1, 4)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        game.move(RobotRebootAction(0, Direction.WEST))
        self.assertEqual([(1, 2)], game.robots)

    def test_is_robot_on(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
        ]))
        robots = [(1, 4), [0, 3], [2, 1]]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        for maze_pos in np.ndenumerate(maze.cells):
            if maze_pos in robots:
                self.assertTrue(game.is_a_robot_on(maze_pos))
            else:
                self.assertFalse(game.is_a_robot_on(maze_pos))

    def test_is_over(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
        ]))
        robots = [(0, 3), [1, 4], [2, 1]]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        self.assertFalse(game.is_over())
        game.move(RobotRebootAction(0, Direction.WEST))
        self.assertTrue(game.is_over())

    def test_is_over_max_out_movements(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
        ]))
        robots = [(0, 3), [1, 4], [2, 1]]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)), max_movements=2)
        [game.move(RobotRebootAction(2, Direction.SOUTH)) for i in range(game.max_movements + 1)]
        self.assertTrue(game.is_over())

    def test_score(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
        ]))
        robots = [(0, 3), [1, 4], [2, 1]]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)), max_movements=20)
        game.move(RobotRebootAction(0, Direction.WEST))
        self.assertEqual(game.max_movements - 1, game.score())

    def test_state_when_game_not_over(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.SOUTH_WALL, Maze.EMPTY, Maze.EAST_WALL, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.NORTH_WALL, Maze.NORTH_WALL, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.SOUTH_WALL, Maze.EMPTY, Maze.EMPTY],
            [Maze.EAST_WALL, Maze.EAST_WALL, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        maze_cells = maze.cells.copy()
        robots = [(0, 2), (0, 2), (4, 2)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(1, (3, 4)))
        obs = game.state()

        rows, cols, layers = obs.shape

        self.assertEqual(obs.shape, (5, 5, 4))
        np.testing.assert_equal(obs[:, :, layers - 1], maze_cells)
        # Checking robots on each layer
        self.assertEqual(obs[0, 2, 0], RobotRebootGame.ROBOT)
        self.assertEqual(obs[0, 2, 1], RobotRebootGame.ROBOT)
        self.assertEqual(obs[4, 2, 2], RobotRebootGame.ROBOT)
        # Checking goal on the target robot
        self.assertEqual(obs[3, 4, 1], RobotRebootGame.GOAL)

        for i in range(len(robots)):
            robot_layer = obs[:, :, i]
            maze_positions_not_empty = np.argwhere(robot_layer != RobotRebootGame.EMPTY)[0]
            robot_pos = np.array(robots[i])
            np.testing.assert_equal(robot_pos, maze_positions_not_empty)

    def test_state_when_game_won(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.SOUTH_WALL, Maze.EMPTY, Maze.SOUTH_WALL, Maze.EMPTY]
        ]))
        robots = [(0, 0)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(0, (0, 0)))
        obs = game.state()
        self.assertEqual(RobotRebootGame.GOAL_ROBOT, obs[0, 0, 0])

    def test_observation(self):
        maze = Maze(np.array([
            [Maze.EMPTY, Maze.SOUTH_WALL, Maze.EMPTY, Maze.EAST_WALL, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.NORTH_WALL, Maze.NORTH_WALL, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY],
            [Maze.EMPTY, Maze.EMPTY, Maze.SOUTH_WALL, Maze.EMPTY, Maze.EMPTY],
            [Maze.EAST_WALL, Maze.EAST_WALL, Maze.EMPTY, Maze.EMPTY, Maze.EMPTY]
        ]))
        robots = [(0, 2), (1, 2), (4, 2)]
        game = RobotRebootGame(maze, robots.copy(), RobotRebootGoal(1, (3, 4)))

        obs = game.observation()

        self.assertEqual(obs.shape, (9, 9, 7))
        empty_layer = np.full((9, 9), RobotRebootGame.EMPTY)
        # First layer is empty TODO: change this
        np.testing.assert_equal(obs[:, :, 0], empty_layer)
        # Robot 0
        robot_0_layer = empty_layer.copy()
        robot_0_layer[0, 4] = RobotRebootGame.PRESENCE
        np.testing.assert_equal(obs[:, :, 1], robot_0_layer)
        np.testing.assert_equal(obs[:, :, 2], empty_layer)

        # Robot 1
        robot_1_layer = empty_layer.copy()
        robot_1_layer[2, 4] = RobotRebootGame.PRESENCE
        robot_1_goal_layer = empty_layer.copy()
        robot_1_goal_layer[6, 8] = RobotRebootGame.PRESENCE
        np.testing.assert_equal(obs[:, :, 3], robot_1_layer)
        np.testing.assert_equal(obs[:, :, 4], robot_1_goal_layer)

        # Robot 2
        robot_2_layer = empty_layer.copy()
        robot_2_layer[8, 4] = RobotRebootGame.PRESENCE
        np.testing.assert_equal(obs[:, :, 5], robot_2_layer)