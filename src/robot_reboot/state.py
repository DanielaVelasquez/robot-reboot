import numpy as np

from exceptions.robot_reboot.state import EmptyRobotsPositionException, InvalidRobotsPositionException, \
    RobotsPositionOutOfMazeBoundsException, NumberRobotsNotMatchingException, InvalidRobotsList
from exceptions.util import assertOrThrow
from ..alphazero.state import State


class RobotRebootState(State):

    ROBOT_IN_CELL = 1

    """
    State for the robot reboot game is defined by the positions of the robots.
    Attributes:
        robots_positions (list): list of (x,y) values defining where a robot is on the maze
                                 each index in the list represents a robot
    """

    def __init__(self, game, robots_positions, sequence_i=0):
        """ Initializes a robot reboot state
        Args:
            sequence_i       (int):  Moment in time where the state occurred i.e 0  it's how the game started
            game             (Game): Game that the state belongs to
            robots_positions (list): list of (x,y) values defining where a robot is on the maze
                                     each index in the list represents a robot
        """
        assertOrThrow(len(robots_positions) != 0, EmptyRobotsPositionException)
        assertOrThrow(len([rp for rp in robots_positions if rp[0] < 0 or rp[1] < 0]) == 0,
                      InvalidRobotsPositionException())
        assertOrThrow(len([rp for rp in robots_positions if rp[0] >= game.maze.shape[0] or rp[1] >= game.maze.shape[
            1]]) == 0, RobotsPositionOutOfMazeBoundsException())
        assertOrThrow(len(robots_positions) == game.n_robots, NumberRobotsNotMatchingException())
        assertOrThrow(str(type(robots_positions)).__contains__("list"), InvalidRobotsList())
        State.__init__(self, game, sequence_i)
        self.__robots_positions = robots_positions

    @property
    def robots_positions(self):
        return self.__robots_positions

    def is_robot_on(self, pos):
        return len([r for r in self.__robots_positions if r == pos]) != 0

    def __str__(self):
        return f'{self.__robots_positions}'

    def get_state(self):
        r, c = self.game.maze.shape
        n_robots = len(self.__robots_positions)
        maze = np.zeros((r, c, n_robots * 2 + 1))
        maze[:, :, 0] = self.game.maze
        i = 1
        for x, y in self.__robots_positions:
            maze[x, y, i] = RobotRebootState.ROBOT_IN_CELL
            i += 2
        x, y = self.game.goal_house.house
        # Goal house
        maze[x, y, self.game.goal_house.robot_id + 2] = RobotRebootState.ROBOT_IN_CELL
        return maze
