import numpy as np

from exceptions.robot_reboot.game import NoRobotsGameException, InvalidMazeException, \
    RobotHouseOutOfMazeBoundsException, \
    MazeNotSquareException, MazeSizeInvalidException, RobotHouseInvalidRobotIdException
from exceptions.util import assertOrThrow
from src.alphazero.game import Game
from src.robot_reboot.action import RobotRebootAction
from src.robot_reboot.state import RobotRebootState
from .direction import Direction
from .goal_house import RobotRebootGoalHouse
from .maze_cell_type import MazeCellType
from .util import valid_maze


def get_game_from_matrix(matrix):
    rows, cols, layers = matrix.shape
    maze = matrix[:, :, 0]
    robot_house = None
    robots = list()
    for i in range(1, layers):
        if i % 2 != 0:
            rob_pos = np.argwhere(matrix[:, :, i] == RobotRebootState.ROBOT_IN_CELL)
            assert "More than one robot in a robot layer", rob_pos.shape == (1, 2)
            robots.append((rob_pos[0, 0], rob_pos[0, 1]))
        else:
            house_pos = np.argwhere(matrix[:, :, i] == RobotRebootState.ROBOT_IN_CELL)
            is_house = house_pos.shape != (0, 2)
            assert "There is more than one house defined in the matrix", robot_house is not None and is_house
            if is_house:
                robot_house = RobotRebootGoalHouse(int(i / 2) - 1, (house_pos[0, 0], house_pos[0, 1]))

    assert "No robot positions found in the maze", len(robots) > 0
    assert "No robot house was defined", robot_house is not None

    game = RobotRebootGame(int((layers - 1) / 2), maze, robot_house)
    state = RobotRebootState(game, robots)
    return game, state


class RobotRebootGame(Game):
    """
    Robot Reboot game, its maze and number of robots to move.
    Attributes:
         n_robots (int):             number of robots in the game
         maze     (np array):        np array with values [-1, 0, 1]. -1 if it is a cell use to represent walls.
                                     1 if there is a wall in that cell. 0 if its a cell where the robots can move.
         goal_house     (RobotRebootGoalHouse): robot that needs to get to its house
    """

    def __init__(self, n_robots, maze, goal_house: RobotRebootGoalHouse):
        """Initializes a robot reboot game
        Args:
            n_robots (int):             number of robots in the game
            maze     (np array):        np array with values [-1, 0, 1]. -1 if it is a cell use to represent walls.
                                        1 if there is a wall in that cell. 0 if its a cell where the robots can move.
            goal_house     (RobotRebootGoalHouse): robot that needs to get to its house
        """
        assertOrThrow(n_robots > 0, NoRobotsGameException())
        assertOrThrow(valid_maze(n_robots, maze), InvalidMazeException())
        assertOrThrow(goal_house.house[0] < maze.shape[0] and goal_house.house[1] < maze.shape[1],
                      RobotHouseOutOfMazeBoundsException())
        assertOrThrow(goal_house.robot_id < n_robots, RobotHouseInvalidRobotIdException())
        assertOrThrow(maze.shape[0] == maze.shape[1], MazeNotSquareException())
        assertOrThrow(maze.shape[0] % 2 != 0, MazeSizeInvalidException())
        Game.__init__(self, [RobotRebootAction(r, d) for r in range(n_robots) for d in Direction])
        self.__n_robots = n_robots
        self.__maze = maze
        self.__goal_house = goal_house

    @property
    def n_robots(self):
        return self.__n_robots

    @property
    def maze(self):
        return self.__maze

    @property
    def goal_house(self):
        return self.__goal_house

    def get_value(self, state: RobotRebootState):
        if state.robots_positions[self.__goal_house.robot_id] == self.__goal_house.house:
            return 1
        return 0

    def is_over(self, state: RobotRebootState):
        return state.robots_positions[self.__goal_house.robot_id] == self.__goal_house.house

    def get_score(self, state: RobotRebootState):
        return state.sequence_i * -1

    def apply(self, action: RobotRebootAction, state: RobotRebootState):
        pos = state.robots_positions[action.robot_id]
        robot_x, robot_y = pos
        rows, cols = self.__maze.shape
        if action.direction == Direction.NORTH:
            walls = np.argwhere(self.__maze[:robot_x, robot_y] == MazeCellType.WALL.value)
            if walls.size == 0:
                robots_positions = state.robots_positions.copy()
                robots_positions[action.robot_id] = (0, robot_y)
                return RobotRebootState(self, robots_positions, state.sequence_i + 1)
            else:
                new_x = walls[walls.size - 1][0] + 1
                new_pos = (new_x, robot_y)
                if state.is_robot_on(new_pos) and new_pos != pos:
                    new_x += 2
                robots_positions = state.robots_positions.copy()
                robots_positions[action.robot_id] = (new_x, robot_y)
                return RobotRebootState(self, robots_positions, state.sequence_i + 1)
        elif action.direction == Direction.SOUTH:
            walls = np.argwhere(self.__maze[robot_x + 1:, robot_y] == MazeCellType.WALL.value)
            if walls.size == 0:
                robots_positions = state.robots_positions.copy()
                robots_positions[action.robot_id] = (rows - 1, robot_y)
                return RobotRebootState(self, robots_positions, state.sequence_i + 1)
            else:
                new_x = walls[0][0] + robot_x
                new_pos = (new_x, robot_y)
                if state.is_robot_on((new_x, robot_y)) and new_pos != pos:
                    new_x -= 2
                robots_positions = state.robots_positions.copy()
                robots_positions[action.robot_id] = (new_x, robot_y)
                return RobotRebootState(self, robots_positions, state.sequence_i + 1)
        elif action.direction == Direction.WEST:
            walls = np.argwhere(self.__maze[robot_x, :robot_y] == MazeCellType.WALL.value)
            if walls.size == 0:
                robots_positions = state.robots_positions.copy()
                robots_positions[action.robot_id] = (robot_x, 0)
                return RobotRebootState(self, robots_positions, state.sequence_i + 1)
            else:
                new_y = walls[walls.size - 1][0] + 1
                new_pos = (robot_x, new_y)
                if state.is_robot_on(new_pos) and new_pos != pos:
                    new_y += 2
                robots_positions = state.robots_positions.copy()
                robots_positions[action.robot_id] = (robot_x, new_y)
                return RobotRebootState(self, robots_positions, state.sequence_i + 1)
        elif action.direction == Direction.EAST:
            walls = np.argwhere(self.__maze[robot_x, robot_y + 1:] == MazeCellType.WALL.value)
            if walls.size == 0:
                robots_positions = state.robots_positions.copy()
                robots_positions[action.robot_id] = (robot_x, cols - 1)
                return RobotRebootState(self, robots_positions, state.sequence_i + 1)
            else:
                new_y = walls[0][0] + robot_y
                new_pos = (robot_x, new_y)
                if state.is_robot_on(new_pos) and new_pos != pos:
                    new_y -= 2
                robots_positions = state.robots_positions.copy()
                robots_positions[action.robot_id] = (robot_x, new_y)
                return RobotRebootState(self, robots_positions, state.sequence_i + 1)
        else:
            raise Exception("Unsupported direction")

    def get_valid_actions(self, state: RobotRebootState):
        valid_actions = list()
        for action in self.actions:
            if not self.__is_wall_at(state.robots_positions[action.robot_id], action.direction):
                valid_actions.append(action)
        return valid_actions

    def __is_wall_at(self, position: tuple, direction: Direction):
        """Determines if there is an immediate wall in the direction of a position, ie. is there a wall at north of (1,1)
        if the position is at the edge of the maze, then movement outside the maze bounds is considered a wall, ie. if
        there is a wall at north of (0,0) it returns true.
        Args:
            position (tuple): reference position to check a direction with
            direction (Direction): direction to check from the position
        Returns:
            boolean: True if there is wall at the immediate direction or the immediate  position in that direction is out
                     of the maze. False otherwise.
        """
        x, y = position
        rows, cols = self.__maze.shape

        return (direction == Direction.NORTH and (
                (x - 1 >= 0 and self.__maze[x - 1, y] == MazeCellType.WALL.value) or (x - 1 < 0))
                ) or (
                       direction == Direction.SOUTH and (
                       (x + 1 < rows and self.__maze[x + 1, y] == MazeCellType.WALL.value) or (x + 1 >= rows))
               ) or (
                       direction == Direction.WEST and (
                       (y - 1 >= 0 and self.__maze[x, y - 1] == MazeCellType.WALL.value) or (y - 1 < 0))
               ) or (
                       direction == Direction.EAST and (
                       (y + 1 < cols and self.__maze[x, y + 1] == MazeCellType.WALL.value) or (y + 1 >= cols)))
