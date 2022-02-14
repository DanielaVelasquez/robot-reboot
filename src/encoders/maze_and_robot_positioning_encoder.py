import numpy as np

from src.encoders.action_encoder import get_map_actions_to_index_and_actions_list
from src.encoders.base import Encoder
from src.robot_reboot.state import RobotRebootState

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
        game = game_state.game
        encoded_state = np.zeros(self.shape())
        encoded_state[:, :, 0] = game_state.game.maze
        self.__encode_robot_positions(encoded_state, game_state)
        self.__encode_goal_house(encoded_state, game)
        self.__encode_robot_future_positions(encoded_state, game_state)

        return encoded_state

    def __encode_robot_future_positions(self, encoded_state, game_state):
        valid_actions = game_state.get_valid_actions()
        for action in valid_actions:
            next_state = valid_actions[action]
            next_robot_positions = next_state.robots_positions
            next_x, next_y = next_robot_positions[action.robot_id]
            robot_next_positions_layer = (action.robot_id * 3) + 3
            encoded_state[next_x, next_y, robot_next_positions_layer] = RobotRebootState.ROBOT_IN_CELL

    def __encode_goal_house(self, encoded_state, game):
        x, y = game.goal_house.house
        encoded_state[x, y, (game.goal_house.robot_id * 3) + 2] = RobotRebootState.ROBOT_IN_CELL

    def __encode_robot_positions(self, encoded_state, game_state):
        i = 1
        for x, y in game_state.robots_positions:
            encoded_state[x, y, i] = RobotRebootState.ROBOT_IN_CELL
            i += 3

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
