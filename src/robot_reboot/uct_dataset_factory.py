import logging

from src.alphazero.game_player import GamePlayer
from src.ml.model import get_model
from src.robot_reboot.factory import RobotRebootFactory
from src.robot_reboot.model import RobotRebootModel
from src.uct.heuristic_function import uct_heuristic_fn
from src.uct.uct import UCT
import numpy as np

logging.getLogger().setLevel(logging.INFO)


class RobotRebootUCTDataSetFactory:
    """Factory to create a data set input based on the configuration
    Attributes:
        game_factory (RobotRebootFactory): robot reboot factory to create samples for a dataset
        maze_size (int): size of the maze (number of rows and cols)
        max_depth (int): Max depth of the tree search
        cnn: neural network leading the mcts
        playouts: number of playouts per simulation
    """

    def __init__(self, maze_size, cnn, max_depth=20, playouts=10):
        self.__game_factory = RobotRebootFactory()
        self.__maze_size = maze_size
        self.__max_dept = max_depth
        self.__cnn = cnn
        self.__playouts = playouts

    @property
    def game_factory(self):
        return self.__game_factory

    @property
    def maze_size(self):
        return self.__maze_size

    @property
    def max_depth(self):
        return self.__max_dept

    @property
    def cnn(self):
        return self.__cnn

    @property
    def playouts(self):
        return self.__playouts

    def create(self, locate_robot_close_goal=False, max_movements=5):
        """Creates a data set example by creating random games, searching on the tree to build the probabilities,
        executes a play of the game and obtains the value if the NN suggestions were followed
        Returns:
            v (int): value of the final game (win,loss or draw)
            p (np array): Probability distribution of wining per action based on the initial state of the game
            s (state): Initial state where the search and play where applied in its matrix form
        """
        logging.info("Starting to create a dataset")
        game, state, quadrants_ids = self.game_factory.create(self.maze_size,
                                                              locate_robot_close_goal=locate_robot_close_goal,
                                                              max_movements=max_movements)
        logging.info("Game created")
        model = RobotRebootModel(game, self.cnn)
        game_player = GamePlayer(model, game)
        logging.info("Search using MCTS")
        uct = UCT(game, self.max_depth, heuristic_fn=uct_heuristic_fn, playouts=self.playouts)
        p = uct.search(state)
        logging.info("Search finished, probabilities calculated")
        logging.info("Playing the game")
        final_state = game_player.play(state, self.max_depth)
        logging.info("Game outcome calculated")
        v = game.get_value(final_state)
        return v, p, state.get_matrix()


if __name__ == "__main__":
    np.random.seed(26)
    cnn = get_model()
    ds_factory = RobotRebootUCTDataSetFactory(31, cnn, max_depth=20, playouts=50)
    v, p, s = ds_factory.create(locate_robot_close_goal=True, max_movements=3)
    print (v)