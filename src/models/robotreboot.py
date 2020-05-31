from .maze import Maze
import numpy as np


class Goal:
    def __init__(self, robot_id, cell):
        self.robot_id = robot_id
        self.cell = cell


class RobotReboot:
    '''
    Movements
    '''
    N = "N"
    E = "E"
    S = "S"
    W = "W"

    GOAL = 100

    def __init__(self, maze, robots={"A": (0, 0)}, goal=Goal("A", (0, 0))):
        self.maze = maze
        self.robots = robots.copy()
        self.robots_initial = robots.copy()
        self.goal = goal

    def add_robot(self, robot_id, robot_position):
        self.robots[robot_id] = robot_position

    def move_robot(self, robot_id, direction):
        '''
        Moves a robot on the maze
            robot_id: robots identifier
            direction: N, S, E, W
        '''
        if direction == self.N:
            self.__move_north(robot_id)
        elif direction == self.S:
            self.__move_south(robot_id)
        elif direction == self.E:
            self.__move_east(robot_id)
        elif direction == self.W:
            self.__move_west(robot_id)
        else:
            raise Exception("Not a valid direction")

    def __move_north(self, robot_id):
        x, y = self.robots[robot_id]
        #Robot is not at the border of the maze
        if x != 0:
            new_x = x
            cells = self.maze.cells[0:x, y]
            current_row = x - 1
            for i in np.nditer(cells[::-1], order='C'):
                if self.is_a_robot_on((current_row, y)) or i == Maze.S:
                    new_x = current_row + 1
                    break
                elif i == Maze.N:
                    new_x = current_row
                    break
                else:
                    new_x = current_row
                current_row -= 1
            self.robots[robot_id] = (new_x, y)

    def __move_south(self, robot_id):
        x, y = self.robots[robot_id]
        if x != self.maze.width - 1:
            new_x = x
            cells = self.maze.cells[x+1:, y]
            current_row = x + 1
            for i in np.nditer(cells, order='C'):
                if self.is_a_robot_on((current_row, y)) or i == Maze.N:
                    new_x = current_row - 1
                    break
                elif i == Maze.S:
                    new_x = current_row
                    break
                else:
                    new_x = current_row
                current_row += 1
            self.robots[robot_id] = (new_x, y)

    def __move_west(self, robot_id):
        x, y = self.robots[robot_id]
        if y != 0:
            new_y = y
            cells = self.maze.cells[x, :y]
            current_col = y - 1
            for i in np.nditer(cells[::-1], order='C'):
                if self.is_a_robot_on((x, current_col)) or i == Maze.W:
                    new_y = current_col + 1
                    break
                elif i == Maze.E:
                    new_y = current_col
                    break
                else:
                    new_y = current_col
                current_col -= 1
            self.robots[robot_id] = (x, new_y)

    def __move_east(self, robot_id):
        x, y = self.robots[robot_id]
        if y != self.maze.height - 1:
            new_y = y
            cells = self.maze.cells[x, y + 1:]
            current_col = y + 1
            for i in np.nditer(cells, order='C'):
                if self.is_a_robot_on((x, current_col)) or i == Maze.W:
                    new_y = current_col - 1
                    break
                elif i == Maze.E:
                    new_y = current_col
                    break
                else:
                    new_y = current_col
                current_col += 1
            self.robots[robot_id] = (x, new_y)

    def is_a_robot_on(self, pos):
        for robot_id in self.robots:
            robot_pos = self.robots[robot_id]
            if robot_pos == pos:
                return True
        return False

    def is_goal_robot(self, robot_id):
        return robot_id == self.goal.robot_id

    def reset(self):
        self.robots = self.robots_initial.copy()

    @property
    def state(self):
        """
        An observation is a three dimensional array.
        First layer represents the maze's structure (i.e its walls)
        For each robot on the game there is an extra layer. Each layer will have a 1 wherever the robot is located
            in the maze. On the goal robot's layer, the goal cell is set to the GOAL value.
        """
        total_robots = len(dict.keys(self.robots))
        rows, cols = self.maze.cells.shape
        obs = np.zeros((rows, cols, total_robots + 1))
        obs[:, :, 0] = self.maze.cells
        layer = 1
        for robot_id in self.robots:
            x, y = self.robots[robot_id]
            obs[x, y, layer] = 1
            if robot_id == self.goal.robot_id:
                x_goal, y_goal = self.goal.cell
                obs[x_goal, y_goal, layer] = self.GOAL
            layer += 1
        return obs

    @property
    def done(self):
        return self.robots[self.goal.robot_id] == self.goal.cell

    @property
    def count_robots(self):
        return len(self.robots)

