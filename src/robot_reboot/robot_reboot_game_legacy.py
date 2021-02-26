import deprecation
import numpy as np

from src.alphazero.game_legacy import GameLegacy, GameActionLegacy
from src.robot_reboot.util import Direction, Maze, calculate_size_with_walls


class RobotRebootActionLegacy(GameActionLegacy):
    def __init__(self, robot_id, movement_direction: Direction):
        self.robot = robot_id
        self.movement_direction = movement_direction
        self.position_before_move = tuple()

    def __eq__(self, other):
        if not isinstance(other, RobotRebootActionLegacy):
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


class RobotRebootGoalLegacy:

    def __init__(self, robot, pos: tuple):
        self.robot = robot
        self.position = pos

    def __str__(self):
        return f'Robot {self.robot} on position {self.position}'


class RobotRebootGameLegacy(GameLegacy):
    MOVEMENTS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]

    GOAL = 9
    ROBOT = 5
    GOAL_ROBOT = 10
    EMPTY = 0
    PRESENCE = 1
    FORBIDDEN = - 1

    def __init__(self, maze: Maze, robots: list, goal: RobotRebootGoalLegacy):
        super().__init__()
        self.maze = maze
        self.robots = robots
        self.goal = goal

    def __add_robot_position_history(self, robot_index, previous_position):
        q = self.robots_position_history[robot_index]
        q.append(previous_position)

    def score(self):
        return 1 if self.robots[self.goal.robot] == self.goal.position else 0

    def win(self):
        return True if self.score() == 1 else False

    def is_over(self):
        return self.robots[self.goal.robot] == self.goal.position

    def get_valid_actions(self):
        return [RobotRebootActionLegacy(robot, movement) for movement in self.MOVEMENTS for robot in
                range(len(self.robots)) if self.can_move(RobotRebootActionLegacy(robot, movement))]

    def get_all_actions(self):
        return [RobotRebootActionLegacy(robot, movement) for movement in self.MOVEMENTS for robot in
                range(len(self.robots))]

    def execute_move(self, action: RobotRebootActionLegacy):
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

    def execute_undo_move(self, action: RobotRebootActionLegacy):
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

    def can_move(self, action: RobotRebootActionLegacy):
        current_pos = self.robots[action.robot]
        self.move(action)
        after_pos = self.robots[action.robot]
        self.undo_move()
        return current_pos != after_pos

    @deprecation.deprecated(deprecated_in="1.0")
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

    def observation(self):
        """
        An observation is a three dimensional array.
        First layer represents the maze's structure (i.e its walls)
        For each robot's position there is a layer follow by a goal layer, that depict, where the robot is aiming to
        get to. If the robot doesn't have a goal in the current game, its goal layer is empty.

        Maze layer
        For each robot
            Robot position layer
            Robot goal layer
        """

        total_robots = len(self.robots)
        rows, cols = self.maze.cells.shape
        assert rows == cols
        new_size = calculate_size_with_walls(rows)
        obs = np.full((new_size, new_size, total_robots * 2 + 1), self.EMPTY)

        # Maze layer
        obs[:, :, 0] = self.__get_maze_layer(new_size)

        index = 1
        for robot_id in range(0, len(self.robots)):
            robot_layer, goal_layer = self.__get_robot_and_robot_goal_layer(new_size, robot_id)
            obs[:, :, index] = robot_layer
            obs[:, :, index + 1] = goal_layer
            index += 2
        return obs

    def __get_maze_layer(self, maze_size):
        maze_layer = np.full((maze_size, maze_size), self.EMPTY)
        walls = [i for i in range(maze_size) if i % 2 != 0]
        maze_layer[walls, :] = self.FORBIDDEN
        maze_layer[:, walls] = self.FORBIDDEN

        rows, cols = self.maze.cells.shape
        for i in range(rows):
            for j in range(cols):
                real_x, real_y = i * 2, j * 2
                wall = self.maze.cells[i, j]
                if wall != Maze.EMPTY:
                    if wall == Maze.NORTH_WALL and real_x - 1 > 0:
                        maze_layer[real_x - 1, real_y] = self.PRESENCE
                    elif wall == Maze.SOUTH_WALL and real_x + 1 < maze_size:
                        maze_layer[real_x + 1, real_y] = self.PRESENCE
                    elif wall == Maze.WEST_WALL and real_y - 1 > 0:
                        maze_layer[real_x, real_y - 1] = self.PRESENCE
                    elif wall == Maze.EAST_WALL and real_y + 1 < maze_size:
                        maze_layer[real_x, real_y + 1] = self.PRESENCE
                    else:
                        raise Exception("Invalid wall value")

        return maze_layer

    def __get_robot_and_robot_goal_layer(self, maze_size, robot_id):
        x, y = self.robots[robot_id]
        robot_layer = np.full((maze_size, maze_size), self.EMPTY)
        robot_layer[x * 2, y * 2] = self.PRESENCE

        goal_layer = np.full((maze_size, maze_size), self.EMPTY)
        if self.goal.robot == robot_id:
            goal_x, goal_y = self.goal.position
            goal_layer[goal_x * 2, goal_y * 2] = self.PRESENCE
        return robot_layer, goal_layer