from src.exceptions.exceptions import RequiredValueException
from src.game.action import Action
from src.exceptions.util import assert_or_throw


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
        assert_or_throw(robot_id is not None, RequiredValueException("robot_id"))
        assert_or_throw(direction is not None, RequiredValueException("direction"))
        self.__robot_id = robot_id
        self.__direction = direction

    @property
    def robot_id(self):
        return self.__robot_id

    @property
    def direction(self):
        return self.__direction

    def __str__(self):
        return f'Moving robot {self.__robot_id} on {self.__direction}'

    def __eq__(self, obj):
        return isinstance(obj, RobotRebootAction) and obj.robot_id == self.__robot_id and obj.direction == self.__direction

    def __hash__(self):
        return hash(tuple((self.robot_id, self.direction)))
