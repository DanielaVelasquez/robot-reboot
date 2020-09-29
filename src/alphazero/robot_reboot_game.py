import numpy as np

from datetime import datetime
from src.alphazero.deep_heuristic import DeepHeuristic
from src.alphazero.game import Game, GameAction
from src.alphazero.util import Direction, Maze


class RobotRebootAction(GameAction):
    def __init__(self, robot_id, movement_direction: Direction):
        self.robot = robot_id
        self.movement_direction = movement_direction
        self.position_before_move = tuple()

    def __eq__(self, other):
        if not isinstance(other, RobotRebootAction):
            return NotImplemented
        return self.robot == other.robot and self.movement_direction == other.movement_direction

    def __hash__(self):
        return self.robot + self.movement_direction.value

    def __str__(self):
        if self.movement_direction == Direction.NORTH:
            direction = "North"
        elif self.movement_direction == Direction.SOUTH:
            direction = "South"
        elif self.movement_direction == Direction.EAST:
            direction = "East"
        elif self.movement_direction == Direction.WEST:
            direction = "West"
        else:
            raise Exception("Missing string value for " + self.movement_direction)
        return f'Move robot {self.robot} in {direction} direction'


class RobotRebootGoal:

    def __init__(self, robot, pos: tuple):
        self.robot = robot
        self.position = pos

    def __str__(self):
        return f'Robot {self.robot} on position {self.position}'


class RobotRebootGame(Game):
    MOVEMENTS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]

    GOAL = 9
    ROBOT = 5
    GOAL_ROBOT = 10
    EMPTY = 0

    def __init__(self, maze: Maze, robots: list, goal: RobotRebootGoal, max_movements=4):
        super().__init__()
        self.maze = maze
        self.robots = robots
        self.goal = goal
        self.max_movements = max_movements

    def __add_robot_position_history(self, robot_index, previous_position):
        q = self.robots_position_history[robot_index]
        q.append(previous_position)

    def score(self):
        return 1 if self.robots[self.goal.robot] == self.goal.position else 0

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
        self.move(action)
        after_pos = self.robots[action.robot]
        self.undo_move()
        return current_pos != after_pos

    def state(self):
        """
        An observation is a three dimensional array.
        First layer represents the maze's structure (i.e its walls)
        For each robot on the game there is an extra layer. Each layer will have a 1 wherever the robot is located
            in the maze. On the goal robot's layer, the goal cell is set to the GOAL value.
        """

        total_robots = len(self.robots)
        rows, cols = self.maze.cells.shape
        obs = np.full((rows, cols, total_robots + 1), self.EMPTY)
        obs[:, :, total_robots] = self.maze.cells
        for robot_id in range(0, len(self.robots)):
            x, y = self.robots[robot_id]
            obs[x, y, robot_id] = self.ROBOT
            if robot_id == self.goal.robot:
                x_goal, y_goal = self.goal.position
                obs[x_goal, y_goal, robot_id] = self.GOAL
                # Game is finished
                if self.robots[robot_id] == self.goal.position:
                    obs[x_goal, y_goal, robot_id] = self.GOAL_ROBOT
        # if normalize:
        #     obs = obs / self.GOAL_ROBOT
        return obs


class RobotRebootConfiguration:
    def __init__(self, maze_size: tuple, robots: int, goals: list, max_movements=20):
        self.maze_size = maze_size
        self.robots = robots
        self.goals = goals
        self.max_movements = max_movements


class RobotRebootFactory:
    __MAZE_8_X_8 = 8

    __CONF = {
        __MAZE_8_X_8: RobotRebootConfiguration((8, 8), 2, [(0, 5), (2, 2), (5, 4), (6, 6)], max_movements=5)
    }

    def __init__(self, size=8, seed=26):
        self.size = size
        np.random.seed(seed)

    def build(self):
        maze = self.__generate_maze()
        goal = RobotRebootGoal(0, (4, 0))  # self.__generate_goal()
        robots = [(6, 0), (1, 6)]  # self.__generate_robots(goal)
        return RobotRebootGame(Maze(maze), robots, goal, self.__CONF[self.size].max_movements)

    def __generate_maze(self):
        if self.size in self.__CONF:
            maze = np.full(self.__CONF[self.size].maze_size, Maze.EMPTY)
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


if __name__ == "__main__":
    factory = RobotRebootFactory(size=8)
    # Generate 10 games
    for i in range(1):
        print(f'Game {i}')
        game = factory.build()
        deep_heuristic = DeepHeuristic((8, 8, 3), len(game.get_all_actions()),
                                       model_name='model.h5')
        deep_heuristic.load_model()
        # Play that game until it is over
        while not game.is_over():
            action = deep_heuristic.best_action(game, training=True)
            print(f'Game {i} with action {action}')
            game.move(action)
        print('Won' if game.score() == 1 else 'Lost')

    print(f'End time: {datetime.now()}')
