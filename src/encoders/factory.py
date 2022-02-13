from src.encoders.maze_and_robot_positioning_encoder import POSITIONING_ENCODER_NAME, \
    MazeAndRobotPositioningEncoderBuilder
from src.encoders.maze_and_two_planes_per_robot import MazeAndTwoPlanesPerRobotBuilder, \
    MAZE_AND_TWO_PLANES_PER_ROBOT_ENCODER_NAME


class EncoderFactory:
    def __init__(self):
        self.__builders = {}

    def register_encoder(self, name, builder):
        self.__builders[name] = builder

    def get_by_name(self, name, **kwargs):
        builder = self.__builders.get(name)
        if not builder:
            raise ValueError("Encoder with name " + name + " not found")
        return builder(**kwargs)


class RobotRebootEncoderFactory:
    def __init__(self, load_default=True):
        self.__encoder_factory = EncoderFactory()
        if load_default:
            self.__encoder_factory.register_encoder(MAZE_AND_TWO_PLANES_PER_ROBOT_ENCODER_NAME,
                                                    MazeAndTwoPlanesPerRobotBuilder())
            self.__encoder_factory.register_encoder(POSITIONING_ENCODER_NAME, MazeAndRobotPositioningEncoderBuilder())

    def register_encoder(self, name, builder):
        self.__encoder_factory.register_encoder(name, builder)

    def get_by_name(self, name, game):
        args = {'game': game}
        return self.__encoder_factory.get_by_name(name, **args)
