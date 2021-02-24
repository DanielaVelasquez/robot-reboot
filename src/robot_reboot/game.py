from src.alphazero.game import Game
from src.robot_reboot.action import RobotRebootAction
from src.robot_reboot.state import RobotRebootState
from src.robot_reboot.util import Direction
from .util import valid_maze


class RobotRebootGoal:

    def __init__(self, robot, pos: tuple):
        self.robot = robot
        self.position = pos

    def __str__(self):
        return f'Robot {self.robot} on position {self.position}'


class RobotRebootGame(Game):
    """
    Robot Reboot game, its maze and number of robots to move.
    Attributes:
         n_robots (int):      number of robots in the game
         maze     (np array): np array with values [-1, 0, 1]. -1 if it is a cell use to represent walls.
                              1 if there is a wall in that cell. 0 if its a cell where the robots can move.
    """

    def __init__(self, n_robots, maze):
        """Initializes a robot reboot game
        Args:
            n_robots (int):      number of robots in the game
            maze     (np array): np array with values [0, 1].
                                 1 if there is a wall in that cell. 0 if its a cell where the robots can move.
        """
        assert n_robots > 0, "n_robots should be greater than zero"
        assert valid_maze(n_robots, maze), "Maze is invalid"
        Game.__init__(self, [RobotRebootAction(r, d) for r in range(n_robots) for d in Direction])
        self.__n_robots = n_robots
        self.__maze = maze

    @property
    def n_robots(self):
        return self.__n_robots

    @property
    def maze(self):
        return self.__maze

    def get_value(self, state: RobotRebootState):
        pass

    def is_over(self, state: RobotRebootState):
        pass

    def get_score(self, state: RobotRebootState):
        pass

    def apply(self, action: RobotRebootAction, state: RobotRebootState):
        pass


