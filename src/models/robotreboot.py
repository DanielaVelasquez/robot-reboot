from models.maze import Maze
import numpy as np

class RobotReboot:

    def __init__(self, maze, robots={}, goal=None):
        self.maze = maze
        self.robots = robots
        self.goal = goal

    def add_robot(self, robot_id, robot_position):
        self.robots[robot_id] = robot_position

    def move_robot(self, robot_id, direction):
        '''
        Moves a robot on the maze
            robot_id: robots identifier
            direction: N, S, E, W
        '''
        if direction == 'N':
            self.__move_north(robot_id)
        elif direction == 'S':
            self.__move_south(robot_id)
        elif direction == 'E':
            self.__move_east(robot_id)
        elif direction == 'W':
            self.__move_west(robot_id)
        else:
            raise Exception("Not valid direction")

    def __move_north(self, robot_id):
        x, y = self.robots[robot_id]
        #Robot is not at the border of the maze
        if x != 0:
            new_x = x
            cells = self.maze.cells[0:x, y]
            current_row = x - 1
            for i in np.nditer(cells[::-1], order='C'):
                if i == Maze.S:
                    new_x = current_row - 1
                    break
                elif i == Maze.N:
                    new_x = current_row
                    break
                current_row -= 1
            self.robots[robot_id] = (new_x, y)

    def __move_south(self, robot_id):
        x,y = self.robots[robot_id]
        if x != self.maze.height - 1:
            new_x = x
            cells = self.maze.cells[x+1:, y]
            current_row = x + 1
            for i in np.nditer(cells, order='C'):
                if i == Maze.S:
                    new_x = current_row
                    break
                elif i == Maze.N:
                    new_x = current_row - 1
                    break
                current_row += 1
            self.robots[robot_id] = (new_x, y)

    def __move_west(self, robot_id):
        x, y = self.robots[robot_id]
        if y != 0:
            new_y = y
            cells = self.maze.cells[x, :y]
            current_col = y - 1
            for i in np.nditer(cells[::-1], order='C'):
                if i == Maze.E:
                    new_y = current_col
                    break
                elif i == Maze.W:
                    new_y = current_col - 1
                    break
                current_col -= 1
            self.robots[robot_id] = (x, new_y)

    def __move_east(self, robot_id):
        x, y = self.robots[robot_id]
        if y != self.maze.width - 1:
            new_y = y
            cells = self.maze.cells[x, y + 1:]
            current_col = y + 1
            for i in np.nditer(cells, order='C'):
                if i == Maze.E:
                    new_y = current_col
                    break
                elif i == Maze.W:
                    new_y = current_col - 1
                    break
                current_col += 1
            self.robots[robot_id] = (x, new_y)


    @property
    def done(self):
        return self.robots[self.goal.get("robot")] == self.goal.get("position")

