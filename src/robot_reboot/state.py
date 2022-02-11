from src.exceptions.robot_reboot.state import EmptyRobotsPositionException, InvalidRobotsPositionException, \
    RobotsPositionOutOfMazeBoundsException, NumberRobotsNotMatchingException, InvalidRobotsList, \
    RobotsPositionsOnWallsPositionsExceptions
from src.exceptions.util import assert_or_throw
from src.game.state import State


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
        assert_or_throw(str(type(robots_positions)).__contains__("list"), InvalidRobotsList())
        assert_or_throw(len(robots_positions) != 0, EmptyRobotsPositionException)
        assert_or_throw(len([rp for rp in robots_positions if rp[0] < 0 or rp[1] < 0]) == 0,
                        InvalidRobotsPositionException())
        assert_or_throw(len([rp for rp in robots_positions if rp[0] % 2 != 0 or rp[1] % 2 != 0]) == 0,
                        RobotsPositionsOnWallsPositionsExceptions())
        assert_or_throw(len([rp for rp in robots_positions if rp[0] >= game.maze.shape[0] or rp[1] >= game.maze.shape[
            1]]) == 0, RobotsPositionOutOfMazeBoundsException())
        assert_or_throw(len(robots_positions) == game.robots_count, NumberRobotsNotMatchingException())
        State.__init__(self, game, sequence_i)
        self.__robots_positions = robots_positions

    @property
    def robots_positions(self):
        return self.__robots_positions

    @property
    def robots_count(self):
        return len(self.__robots_positions)

    def is_robot_on(self, pos):
        return len([r for r in self.__robots_positions if r == pos]) != 0

    def get_valid_actions(self):
        return self.game.get_valid_actions(self)

    def __str__(self):
        return f'{self.__robots_positions}'
