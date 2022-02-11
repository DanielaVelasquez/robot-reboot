import numpy as np

from src.encoders.base import Encoder
from src.robot_reboot.action import RobotRebootAction
from src.robot_reboot.direction import Direction
from src.robot_reboot.state import RobotRebootState


class MazeAndTwoPlanesPerRobot(Encoder):
    """Encodes a game state into a matrix with zeros and ones. It's 3-dimensional array where:
     - The first plane represents the maze and its walls. 1 For a wall and Zero for an empty cell.
     - Next planes, for each robot
        - Robot position. 1 For the robot position, zero otherwise.
        - Robot goal house. 1 For the robot house, zero otherwise
    If the robot goal house doesn't exists, the layer is full of zeros
    """

    def __init__(self, n_robots, maze_size):
        self.__n_robots = n_robots
        self.__maze_size_rows, self.__maze_size_cols = maze_size
        self.__map_actions_to_index, self.__actions_list = self.__get_encoded_decoded_actions()

    def __get_encoded_decoded_actions(self):
        actions = [RobotRebootAction(r, d) for r in range(self.__n_robots) for d in Direction]
        return {actions[i]: i for i in range(len(actions))}, actions

    def name(self):
        return 'maze-and-two-planes-per-robot'

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


class MazeAndTwoPlanesPerRobotBuilder:
    def __call__(self, n_robots, maze_size):
        return MazeAndTwoPlanesPerRobot(n_robots, maze_size)
