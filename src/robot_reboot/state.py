from ..alphazero.state import State


class RobotRebootState(State):
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
        assert len(robots_positions) != 0, "Robots position must be provided"
        State.__init__(self, game, sequence_i)
        self.__robots_positions = robots_positions

    @property
    def robots_positions(self):
        return self.__robots_positions

    def __str__(self):
        return f'{self.__robots_positions}'
