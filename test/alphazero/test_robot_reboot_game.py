import unittest
import numpy as np
from src.alphazero.robot_reboot_game import RobotRebootGame, RobotRebootAction, Maze
from src.util.util import get_movement_same_direction_for_wall
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
                             f'Movement {movement} should NOT be allowed, there is a wall in the way')
