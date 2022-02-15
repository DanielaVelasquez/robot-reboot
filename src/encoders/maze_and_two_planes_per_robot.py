import numpy as np

from src.encoders.action_encoder import get_map_actions_to_index_and_actions_list
from src.encoders.base import Encoder
from src.robot_reboot.state import RobotRebootState

MAZE_AND_TWO_PLANES_PER_ROBOT_ENCODER_NAME = 'maze-and-two-planes-per-robot'


class MazeAndTwoPlanesPerRobotEncoder(Encoder):
    """Encodes a game state into a matrix with zeros and ones. It's 3-dimensional array where:
     - The first plane represents the maze and its walls. 1 For a wall and Zero for an empty cell.
     - Next planes, for each robot
        - Robot position. 1 For the robot position, zero otherwise.
        - Robot goal house. 1 For the robot house, zero otherwise
    If the robot goal house doesn't exists, the layer is full of zeros
    """

    def __init__(self, game):
        self.__n_robots = game.robots_count
        self.__maze_size_rows, self.__maze_size_cols = game.maze_shape
        self.__map_actions_to_index, self.__actions_list = get_map_actions_to_index_and_actions_list(game)

    def name(self):
        return MAZE_AND_TWO_PLANES_PER_ROBOT_ENCODER_NAME

    def encode(self, game_state):
        maze_matrix = np.zeros(self.shape())
        game = game_state.game
        maze_matrix[:, :, 0] = game.maze
        i = 1
        for x, y in game_state.robots_positions:
            maze_matrix[x, y, i] = RobotRebootState.ROBOT_IN_CELL
            i += 2
        x, y = game.goal_house.house
        # Goal house
        maze_matrix[x, y, (game.goal_house.robot_id + 1) * 2] = RobotRebootState.ROBOT_IN_CELL
        return maze_matrix

    def encode_action(self, action):
        return self.__map_actions_to_index[action]

    def decode_action_index(self, index):
        return self.__actions_list[index]

    def shape(self):
        planes = 1 + self.__n_robots * 2
        return self.__maze_size_rows, self.__maze_size_cols, planes

    def num_actions(self):
        return len(self.__actions_list)


class MazeAndTwoPlanesPerRobotBuilder:
    def __call__(self, game):
        return MazeAndTwoPlanesPerRobotEncoder(game)
