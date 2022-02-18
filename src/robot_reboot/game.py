import numpy as np

from src.exceptions.robot_reboot.game import NoRobotsGameException, InvalidMazeException, \
    RobotHouseOutOfMazeBoundsException, \
    MazeNotSquareException, MazeSizeInvalidException, RobotHouseInvalidRobotIdException
from src.exceptions.util import assert_or_throw
from src.game.game import Game
from src.robot_reboot.action import RobotRebootAction
from src.robot_reboot.state import RobotRebootState
from .direction import Direction
from .goal_house import RobotRebootGoalHouse
from .maze_cell_type import MazeCellType
from .util import valid_maze, is_even


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
        assert_or_throw(n_robots > 0, NoRobotsGameException())
        assert_or_throw(valid_maze(n_robots, maze), InvalidMazeException())
        assert_or_throw(goal_house.house[0] < maze.shape[0] and goal_house.house[1] < maze.shape[1],
                        RobotHouseOutOfMazeBoundsException())
        assert_or_throw(goal_house.robot_id < n_robots, RobotHouseInvalidRobotIdException())
        assert_or_throw(maze.shape[0] == maze.shape[1], MazeNotSquareException())
        assert_or_throw(maze.shape[0] % 2 != 0, MazeSizeInvalidException())
        Game.__init__(self, [RobotRebootAction(r, d) for r in range(n_robots) for d in Direction])
        self.__n_robots = n_robots
        self.__maze = maze
        self.__goal_house = goal_house

    @property
    def robots_count(self):
        return self.__n_robots

    @property
    def maze(self):
        return self.__maze

    @property
    def maze_shape(self):
        return self.__maze.shape

    @property
    def goal_house(self):
        return self.__goal_house

    def get_value(self, state: RobotRebootState):
        return 1 if state.robots_positions[self.__goal_house.robot_id] == self.__goal_house.house else 0

    def is_over(self, state: RobotRebootState):
        return state.robots_positions[self.__goal_house.robot_id] == self.__goal_house.house

    def get_score(self, state: RobotRebootState):
        return state.sequence_i * -1

    def apply(self, action: RobotRebootAction, state: RobotRebootState):
        pos = state.robots_positions[action.robot_id]
        robot_x, robot_y = pos
        maze = self.__place_robots_on_maze(state.robots_positions)
        rows, cols = maze.shape
        if action.direction == Direction.North:
            walls = np.argwhere(maze[:robot_x, robot_y] == MazeCellType.WALL.value)
            if walls.size == 0:
                return self.__move_to(action.robot_id, (0, robot_y), state)
            else:
                new_x = walls[walls.size - 1][0] + 1
                if not is_even(new_x):
                    new_x += 1
                return self.__move_to(action.robot_id, (new_x, robot_y), state)
        elif action.direction == Direction.South:
            walls = np.argwhere(maze[robot_x + 1:, robot_y] == MazeCellType.WALL.value)
            if walls.size == 0:
                return self.__move_to(action.robot_id, (rows - 1, robot_y), state)
            else:
                new_x = walls[0][0] + robot_x
                if not is_even(new_x):
                    new_x -= 1
                return self.__move_to(action.robot_id, (new_x, robot_y), state)
        elif action.direction == Direction.West:
            walls = np.argwhere(maze[robot_x, :robot_y] == MazeCellType.WALL.value)
            if walls.size == 0:
                return self.__move_to(action.robot_id, (robot_x, 0), state)
            else:
                new_y = walls[walls.size - 1][0] + 1
                if not is_even(new_y):
                    new_y += 1
                return self.__move_to(action.robot_id, (robot_x, new_y), state)
        elif action.direction == Direction.East:
            walls = np.argwhere(maze[robot_x, robot_y + 1:] == MazeCellType.WALL.value)
            if walls.size == 0:
                return self.__move_to(action.robot_id, (robot_x, cols - 1), state)
            else:
                new_y = walls[0][0] + robot_y
                if not is_even(new_y):
                    new_y -= 1
                return self.__move_to(action.robot_id, (robot_x, new_y), state)
        else:
            raise Exception("Unsupported direction")

    def __place_robots_on_maze(self, robots_positions):
        maze = self.__maze.copy()
        for x, y in robots_positions:
            maze[x, y] = MazeCellType.WALL.value
        return maze

    def __move_to(self, robot_id, new_pos, state: RobotRebootState):
        robots_positions = state.robots_positions.copy()
        robots_positions[robot_id] = new_pos
        return RobotRebootState(self, robots_positions, state.sequence_i + 1, previous_state=state,
                                zobrist_hash_generator=state.zobrist_hash_generator)

    def get_valid_actions_next_state_map(self, state: RobotRebootState):
        valid_actions = {}
        for action in self.actions:

            next_state = self.apply(action, state)
            next_robot_positions = next_state.robots_positions
            moved_to_goal_house = next_state.robots_positions[action.robot_id] == self.goal_house.house
            is_correct_robot_goal = action.robot_id is self.goal_house.robot_id
            if state.robots_positions != next_robot_positions and \
                    next_state.zobrist_hash not in state.previous_states:
                if moved_to_goal_house:
                    if is_correct_robot_goal:
                        valid_actions[action] = next_state
                else:
                    valid_actions[action] = next_state

        return valid_actions
