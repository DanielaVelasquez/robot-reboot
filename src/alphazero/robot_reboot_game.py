import numpy as np

from .game import Game, GameAction
from src.alphazero.util import Direction, Maze, get_opposite_direction


class RobotRebootAction(GameAction):
    def __init__(self, robot_id, movement_direction: Direction):
        self.robot = robot_id
        self.movement_direction = movement_direction
        self.position_before_move = tuple()

    def __eq__(self, other):
        if not isinstance(other, RobotRebootAction):
            return NotImplemented
        return self.robot == other.robot and self.movement_direction == other.movement_direction


class RobotRebootGoal:

    def __init__(self, robot, pos: tuple):
        self.robot = robot
        self.position = pos


class RobotRebootGame(Game):
    MOVEMENTS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]

    def __init__(self, maze: Maze, robots: list, goal: RobotRebootGoal, max_movements=20):
        self.maze = maze
        self.robots = robots
        self.goal = goal
        self.max_movements = max_movements

    def __add_robot_position_history(self, robot_index, previous_position):
        q = self.robots_position_history[robot_index]
        q.append(previous_position)

    def score(self):
        return self.max_movements - self.movements

    def is_over(self):
        return self.actions.qsize() > self.max_movements or self.robots[self.goal.robot] == self.goal.position

    def get_valid_actions(self):
        return [RobotRebootAction(robot, movement) for movement in self.MOVEMENTS for robot in
                range(len(self.robots)) if self.can_move(RobotRebootAction(robot, movement))]

    def get_all_actions(self):
        return [RobotRebootAction(robot, movement) for movement in self.MOVEMENTS for robot in
                range(len(self.robots))]

    def execute_move(self, action: RobotRebootAction):
        current_position = self.robots[action.robot]
        action.__setattr__("position_before_move", current_position)
        if action.movement_direction == Direction.NORTH:
            self.__move_north(action.robot)
        elif action.movement_direction == Direction.SOUTH:
            self.__move_south(action.robot)
        elif action.movement_direction == Direction.EAST:
            self.__move_east(action.robot)
        elif action.movement_direction == Direction.WEST:
            self.__move_west(action.robot)
        else:
            raise Exception("Invalid movement")

    def execute_undo_move(self, action: RobotRebootAction):
        self.robots[action.robot] = action.__getattribute__("position_before_move")

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
