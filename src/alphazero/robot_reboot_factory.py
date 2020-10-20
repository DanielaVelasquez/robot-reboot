from abc import ABC

import numpy as np
from src.alphazero.robot_reboot_game import RobotRebootGame, RobotRebootGoal
from src.alphazero.game_factory import GameFactory
from src.alphazero.util import Maze


class RobotRebootConfiguration:
    def __init__(self, maze_size: tuple, robots: int, goals: list):
        self.maze_size = maze_size
        self.robots = robots
        self.goals = goals


class RobotRebootFactory(GameFactory):
    __MAZE_5_X_5 = 5
    __MAZE_8_X_8 = 8

    __CONF = {
        __MAZE_5_X_5: RobotRebootConfiguration((5, 5), 2, [(0, 0), (0, 4), (4, 0), (4, 4)]),
        __MAZE_8_X_8: RobotRebootConfiguration((8, 8), 2, [(0, 5), (2, 2), (5, 4), (6, 6)])
    }

    def __init__(self, size=8, seed=26):
        self.size = size
        np.random.seed(seed)

    def build(self):
        maze = self.__generate_maze()
        goal = self.__generate_goal()
        robots = self.__generate_robots(goal)
        return RobotRebootGame(Maze(maze), robots, goal)

    def build_with(self, seed):
        np.random.seed(seed)
        return self.build()

    def __generate_maze(self):
        if self.size in self.__CONF:
            maze = np.full(self.__CONF[self.size].maze_size, Maze.EMPTY)
            if self.size == self.__MAZE_8_X_8:
                maze[0, 5] = Maze.SOUTH_WALL
                maze[0, 6] = Maze.WEST_WALL
                maze[2, 1] = Maze.EAST_WALL
                maze[2, 2] = Maze.NORTH_WALL
                maze[3, 0] = Maze.SOUTH_WALL
                maze[5, 3] = Maze.EAST_WALL
                maze[5, 4] = Maze.NORTH_WALL
                maze[6, 5] = Maze.EAST_WALL
                maze[6, 6] = Maze.SOUTH_WALL
            return maze
        else:
            raise Exception(f'No defined configuration for size {self.size}')

    def __generate_goal(self):
        if self.size in self.__CONF:
            goals = self.__CONF[self.size].goals
            selected_goal = np.random.randint(len(goals))
            robot = np.random.randint(self.__CONF[self.size].robots)
            return RobotRebootGoal(robot, goals[selected_goal])
        else:
            raise Exception(f'No defined configuration for size {self.size}')

    def __generate_robots(self, goal: RobotRebootGoal):
        if self.size in self.__CONF:
            robots = list()
            for i in range(self.__CONF[self.size].robots):
                x, y = np.random.randint(self.size), np.random.randint(self.size)
                while (x, y) == goal.position and i == goal.robot:
                    x, y = np.random.randint(self.size), np.random.randint(self.size)
                robots.append((x, y))
            return robots
        else:
            raise Exception(f'No defined configuration for size {self.size}')
