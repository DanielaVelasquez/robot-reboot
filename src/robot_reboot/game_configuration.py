class RobotRebootGameConfiguration:
    """A game configuration is a set up of a quadrant of the maze and possible goals to achieve in the quadrant by a robot
    Attributes:
        quadrant (np array):                    2-dimensional matrix tha represents a quadrant of the maze
        houses    (list RobotRebootGoalHouse): list of houses in the quadrant for robots
    """

    def __init__(self, quadrant, houses):
        self.__quadrant = quadrant
        self.__houses = houses

    @property
    def quadrant(self):
        return self.__quadrant

    @property
    def houses(self):
        return self.__houses
