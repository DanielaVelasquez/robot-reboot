import queue
import numpy as np

from .maze import Maze
from .util import copy_queue


class Goal:
    def __init__(self, robot_id, cell):
        self.robot_id = robot_id
        self.cell = cell


class GameState:
    def __init__(self, state):
        self.initial_state = state
        self.movements = list()

    def add_movement(self, robot_id, direction):
        self.movements.append((robot_id, direction))


class RobotReboot:
    '''
    Movements
    '''
    MOVE_NORTH = "N"
    MOVE_EAST = "E"
    MOVE_SOUTH = "S"
    MOVE_WEST = "W"

    GOAL = 9
    ROBOT = 5
    GOAL_ROBOT = 10

    def __init__(self, maze, goals):
        self.maze = maze
        self.goals = copy_queue(goals)
        self.goals_initial = copy_queue(goals)

        self.goal = {}
        self.robots = {}
        self.games = queue.LifoQueue()

        self.__locate_goals()
        self.__start_game()

    def set_game(self, robots, state: np.ndarray, movements):
        _, _, total_robots = state.shape
        assert total_robots - 1 == len(robots)

        self.robots = {}
        self.goal = {}
        self.goals = queue.Queue()
        self.maze = Maze(state[:, :, 0])

        robot_layer = 1
        for robot in robots:
            robot_maze = state[:, :, robot_layer]
            x, y = np.where(robot_maze == self.ROBOT)
            self.robots[robot] = (x[0], y[0])
            goal_x, goal_y = np.where(robot_maze == self.GOAL)
            if goal_x.size != 0 and goal_y.size != 0:
                self.goals.put(Goal(robot, (goal_x[0], goal_y[0])))
            robot_layer += 1

        self.goals_initial = copy_queue(self.goals)
        self.games = queue.LifoQueue()
        self.games.put(GameState(state))

        self.__locate_goals()
        self.goal = self.goals.get()

        for robot_id, direction in movements:
            self.current_game.add_movement(robot_id, direction)
        self.robots_initial = self.robots.copy()

    def __start_game(self):
        """
        Randomize robots on the maze
        """
        for goal in self.goals.queue:
            self.robots[goal.robot_id] = self.__get_random_position_robot()
        self.robots_initial = self.robots.copy()
        self.next_round()

    def __get_random_position_robot(self):
        x = np.random.randint(0, self.maze.width, size=1)[0]
        y = np.random.randint(0, self.maze.height, size=1)[0]

        while (x, y) in self.location_goals or self.is_a_robot_on((x, y)):
            x = np.random.randint(0, self.maze.width, size=1)[0]
            y = np.random.randint(0, self.maze.height, size=1)[0]

        return x, y

    def __locate_goals(self):
        self.location_goals = list()
        for goal in self.goals.queue:
            self.location_goals.append(goal.cell)

    def next_round(self):
        if self.goals.empty():
            self.goals = copy_queue(self.goals_initial)
        self.goal = self.goals.get()
        self.games.put(GameState(self.state))

    def add_robot(self, robot_id, robot_position):
        self.robots[robot_id] = robot_position

    def move_robot(self, robot_id, direction):
        """
        Moves a robot on the maze
            robot_id: robots identifier
            direction: N, S, E, W
        """
        current_game = self.games.get()
        current_game.add_movement(robot_id, direction)
        self.games.put(current_game)

        if direction == self.MOVE_NORTH:
            self.__move_north(robot_id)
        elif direction == self.MOVE_SOUTH:
            self.__move_south(robot_id)
        elif direction == self.MOVE_EAST:
            self.__move_east(robot_id)
        elif direction == self.MOVE_WEST:
            self.__move_west(robot_id)
        else:
            raise Exception("Not a valid direction")

    def __move_north(self, robot_id):
        x, y = self.robots[robot_id]
        if self.maze.cells[x, y] == Maze.N:
            return
            # Robot is not at the border of the maze
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
        if self.maze.cells[x, y] == Maze.S:
            return
        if x != self.maze.width - 1:
            new_x = x
            cells = self.maze.cells[x + 1:, y]
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
        if self.maze.cells[x, y] == Maze.W:
            return
        if y != 0:
            new_y = y
            cells = self.maze.cells[x, :y]
            current_col = y - 1
            for i in np.nditer(cells[::-1], order='C'):
                if self.is_a_robot_on((x, current_col)) or i == Maze.E:
                    new_y = current_col + 1
                    break
                elif i == Maze.W:
                    new_y = current_col
                    break
                else:
                    new_y = current_col
                current_col -= 1
            self.robots[robot_id] = (x, new_y)

    def __move_east(self, robot_id):
        x, y = self.robots[robot_id]
        if self.maze.cells[x, y] == Maze.E:
            return
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

    def is_there_a_robot_except_me(self, pos, robot_id):
        x, y = pos
        return self.is_a_robot_on((x, y)) and self.robots[robot_id] != (x, y)

    def is_goal_robot(self, robot_id):
        return robot_id == self.goal.robot_id

    def reset(self):
        self.robots = self.robots_initial.copy()
        self.goals = copy_queue(self.goals_initial)
        self.goal = self.goals.get()

    def add_robot(self, robot_id, robot_position):
        if robot_id not in self.robots:
            self.robots[robot_id] = robot_position

    def add_robots(self, robots):
        for robot_id in robots:
            self.add_robot(robot_id, robots[robot_id])

    def set_robots(self, robots):
        for robot_id in robots:
            if robot_id in self.robots:
                self.robots[robot_id] = robots[robot_id]

    def state(self, normalize=False):
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
            obs[x, y, layer] = self.ROBOT
            if robot_id == self.goal.robot_id:
                x_goal, y_goal = self.goal.cell
                obs[x_goal, y_goal, layer] = self.GOAL
                # Game is finished
                if self.robots[robot_id] == self.goal.cell:
                    obs[x_goal, y_goal, layer] = self.GOAL_ROBOT
            layer += 1
        if normalize:
            obs = obs/self.GOAL_ROBOT
        return obs

    @property
    def done(self):
        return self.robots[self.goal.robot_id] == self.goal.cell

    @property
    def count_robots(self):
        return len(self.robots)

    @property
    def current_game(self):
        current_game = self.games.get()
        self.games.put(current_game)
        return current_game
