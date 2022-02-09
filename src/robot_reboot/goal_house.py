from src.exceptions.robot_reboot.goals import InvalidRobotIdException
from src.exceptions.util import assertOrThrow


class RobotRebootGoalHouse:
    """Goal of a robot reboot game, taking a robot to its house that is in a position of the maze
    Attributes:
        robot_id (int):   robot identifier that needs to get home
        house    (tuple): position (x,y) where the robot's house is located in the maze
    """

    def __init__(self, robot_id, house: tuple):
        """Initializes a Goal house
        Args:
           robot_id (int):   robot identifier that needs to get home
           house    (tuple): position (x,y) where the robot's house is located in the maze
        """
        assertOrThrow(robot_id >= 0, InvalidRobotIdException())
        self.__robot_id = robot_id
        self.__house = house

    @property
    def robot_id(self):
        return self.__robot_id

    @property
    def house(self):
        return self.__house

    def __str__(self):
        return f'Robot {self.robot_id} needs to get to its house on {self.house}'
