from src.alphazero.game import Game
from src.robot_reboot.action import RobotRebootAction
from src.robot_reboot.state import RobotRebootState
from src.robot_reboot.util import Direction
from .goal_house import RobotRebootGoalHouse
from .util import valid_maze


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
        assert n_robots > 0, "n_robots should be greater than zero"
        assert valid_maze(n_robots, maze), "Maze is invalid"
        assert goal_house.house[0] < maze.shape[0], "goal house row out of the maze"
        assert goal_house.house[1] < maze.shape[1], "goal house column out of the maze"
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
        pass

    def apply(self, action: RobotRebootAction, state: RobotRebootState):
        pass
