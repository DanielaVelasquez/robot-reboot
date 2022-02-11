from src.encoders.maze_and_two_planes_per_robot import MazeAndTwoPlanesPerRobotBuilder


class EncoderFactory:
    def __init__(self):
        self.__builders = {}

    def register_encoder(self, name, builder):
        self.__builders[name] = builder

    def get_by_name(self, name, **kwargs):
        builder = self.__builders[name]
        if not builder:
            raise ValueError("Encoder with name " + name + " not found")
        return builder(**kwargs)


class RobotRebootEncoderFactory:
    def __init__(self, load_default=True):
        self.__encoder_factory = EncoderFactory()
        if load_default:
            self.__encoder_factory.register_encoder('maze-and-two-planes-per-robot', MazeAndTwoPlanesPerRobotBuilder())

    def register_encoder(self, name, builder):
        self.__encoder_factory.register_encoder(name, builder)

    def get_by_name(self, name, n_robots, maze_size):
        args = {'n_robots': n_robots, 'maze_size': maze_size}
        return self.__encoder_factory.get_by_name(name, **args)
