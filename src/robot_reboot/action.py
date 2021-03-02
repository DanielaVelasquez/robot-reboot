from .exceptions import RequiredValueException
from ..alphazero.action import Action
from ..util.util import assertOrThrow


class RobotRebootAction(Action):
    """
    Action for the robot reboot game.
    Attributes:
        robot_id  (int):       robot's id to move
        direction (Direction): direction where the robot is moving (N, S, E, W)
    """

    def __init__(self, robot_id, direction):
        """
        Initializes a Robot reboot action
        Args:
            robot_id  (int):       robot's id to move
            direction (Direction): direction where the robot is moving (N, S, E, W)
        """
        assertOrThrow(robot_id is not None, RequiredValueException("robot_id"))
        assertOrThrow(direction is not None, RequiredValueException("direction"))
        self.__robot_id = robot_id
        self.__direction = direction

    @property
    def robot_id(self):
        return self.__robot_id

    @property
    def direction(self):
        return self.__direction
