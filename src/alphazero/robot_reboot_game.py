import numpy as np

from enum import Enum
from .game import Game, GameAction


class RobotRebootAction(GameAction):
    def __init__(self, robot, movement):
        self.robot = robot
        self.movement = movement


class RobotRebootGoal:

    def __init__(self, robot, pos: tuple):
        self.robot = robot
        self.position = pos


class Maze:
    NORTH_WALL = 1
    EAST_WALL = 2
    SOUTH_WALL = 4
    WEST_WALL = 8
    EMPTY = 0

    WALLS = [NORTH_WALL, EAST_WALL, SOUTH_WALL, WEST_WALL]

    def __init__(self, cells: np.array):
        self.cells = cells
        self.size = cells.shape

    @property
    def height(self):
        return int(self.size[1])

    @property
    def width(self):
        return int(self.size[0])



class RobotRebootGame(Game):
    NORTH_MOVE = 'N'
    EAST_MOVE = 'E'
    SOUTH_MOVE = 'S'
    WEST_MOVE = 'W'

    MOVEMENTS = [NORTH_MOVE, EAST_MOVE, SOUTH_MOVE, WEST_MOVE]

    def __init__(self, maze: Maze, robots: list, goal: RobotRebootGoal, max_movements=20):
        self.maze = maze
        self.robots = robots
        self.goal = goal
        self.max_movements = max_movements

    def score(self):
        return self.max_movements - self.movements

    def is_over(self):
        return self.actions.qsize() > self.max_movements or self.robots[self.goal.robot] == self.goal.position

    def get_valid_actions(self):
        return [RobotRebootAction(robot, movement) for movement in self.__MOVEMENTS for robot in
                range(len(self.robots)) if self.can_move(RobotRebootAction(robot, movement))]

    def get_all_actions(self):
        return [RobotRebootAction(robot, movement) for movement in self.__MOVEMENTS for robot in
                range(len(self.robots))]

    def execute_move(self, action: RobotRebootAction):
        if action.movement == self.NORTH_MOVE:
            self.__move_north(action.robot)
        elif action.movement == self.SOUTH_MOVE:
            self.__move_south(action.robot)
        elif action.movement == self.EAST_MOVE:
            self.__move_east(action.robot)
        elif action.movement == self.WEST_MOVE:
            self.__move_west(action.robot)
        else:
            raise Exception("Invalid movement")

    def execute_undo_move(self, action: RobotRebootAction):
        if action.movement == self.NORTH_MOVE:
            self.__move_south(action.robot)
        elif action.movement == self.SOUTH_MOVE:
            self.__move_north(action.robot)
        elif action.movement == self.EAST_MOVE:
            self.__move_west(action.robot)
        elif action.movement == self.WEST_MOVE:
            self.__move_east(action.robot)
        else:
            raise Exception("Invalid movement")

    def __move_north(self, robot):
        x, y = self.robots[robot]
        if self.maze.cells[x, y] == Maze.NORTH_WALL:
            return
            # Robot is not at the border of the maze
        if x != 0:
            new_x = x
            cells = self.maze.cells[0:x, y]
            current_row = x - 1
            for i in np.nditer(cells[::-1], order='C'):
                if self.is_a_robot_on((current_row, y)) or i == Maze.SOUTH_WALL:
                    new_x = current_row + 1
                    break
                elif i == Maze.NORTH_WALL:
                    new_x = current_row
                    break
                else:
                    new_x = current_row
                current_row -= 1
            self.robots[robot] = (new_x, y)

    def __move_south(self, robot):
        x, y = self.robots[robot]
        if self.maze.cells[x, y] == Maze.SOUTH_WALL:
            return
        if x != self.maze.width - 1:
            new_x = x
            cells = self.maze.cells[x + 1:, y]
            current_row = x + 1
            for i in np.nditer(cells, order='C'):
                if self.is_a_robot_on((current_row, y)) or i == Maze.NORTH_WALL:
                    new_x = current_row - 1
                    break
                elif i == Maze.SOUTH_WALL:
                    new_x = current_row
                    break
                else:
                    new_x = current_row
                current_row += 1
            self.robots[robot] = (new_x, y)

    def __move_west(self, robot):
        x, y = self.robots[robot]
        if self.maze.cells[x, y] == Maze.WEST_WALL:
            return
        if y != 0:
            new_y = y
            cells = self.maze.cells[x, :y]
            current_col = y - 1
            for i in np.nditer(cells[::-1], order='C'):
                if self.is_a_robot_on((x, current_col)) or i == Maze.EAST_WALL:
                    new_y = current_col + 1
                    break
                elif i == Maze.WEST_WALL:
                    new_y = current_col
                    break
                else:
                    new_y = current_col
                current_col -= 1
            self.robots[robot] = (x, new_y)

    def __move_east(self, robot):
        x, y = self.robots[robot]
        if self.maze.cells[x, y] == Maze.EAST_WALL:
            return
        if y != self.maze.height - 1:
            new_y = y
            cells = self.maze.cells[x, y + 1:]
            current_col = y + 1
            for i in np.nditer(cells, order='C'):
                if self.is_a_robot_on((x, current_col)) or i == Maze.WEST_WALL:
                    new_y = current_col - 1
                    break
                elif i == Maze.EAST_WALL:
                    new_y = current_col
                    break
                else:
                    new_y = current_col
                current_col += 1
            self.robots[robot] = (x, new_y)

    def is_a_robot_on(self, pos: tuple):
        """Determines if there is a robot on the maze

       Args:
            pos: Position to check on the maze

        Return:
            boolean: True if there is a robot in give position, false otherwise
        """
        for robot_pos in self.robots:
            if robot_pos == pos:
                return True
        return False

    def can_move(self, action: RobotRebootAction):
        current_pos = self.robots[action.robot]
        self.execute_move(action)
        after_pos = self.robots[action.robot]
        self.execute_undo_move(action)
        return current_pos != after_pos
