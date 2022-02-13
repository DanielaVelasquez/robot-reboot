from src.encoders.action_encoder import get_map_actions_to_index_and_actions_list
from src.encoders.base import Encoder

POSITIONING_ENCODER_NAME = 'maze-and-robot-positioning-encoder'


class MazeAndRobotPositioningEncoder(Encoder):
    """Encodes a game state into a matrix with zeros and ones. It's 3-dimensional array where:
         - The first plane represents the maze and its walls. 1 For a wall and Zero for an empty cell.
         - Next planes, for each robot
            - Robot position. 1 For the robot position, zero otherwise.
            - Robot goal house. 1 For the robot house, zero otherwise.
            - Robot possible next positions.
        If the robot goal house doesn't exists, the layer is full of zeros
        """

    def __init__(self, game):
        self.__n_robots = game.robots_count
        self.__maze_size_rows, self.__maze_size_cols = game.maze_shape
        self.__map_actions_to_index, self.__actions_list = get_map_actions_to_index_and_actions_list(game)

    def name(self):
        return POSITIONING_ENCODER_NAME

    def encode(self, game_state):
        pass

    def encode_action(self, action):
        return self.__map_actions_to_index[action]

    def decode_action_index(self, index):
        return self.__actions_list[index]

    def shape(self):
        planes = 1 + self.__n_robots * 3
        return self.__maze_size_rows, self.__maze_size_cols, planes


class MazeAndRobotPositioningEncoderBuilder:
    def __call__(self, game):
        return MazeAndRobotPositioningEncoder(game)
