from src.alphazero.game_player import GamePlayer
from src.alphazero.heuristic_function import heuristic_fn
from src.alphazero.montecarlo_tree_search import MonteCarloTreeSearch
from src.robot_reboot.factory import RobotRebootFactory
from src.robot_reboot.model import RobotRebootModel


class RobotRebootDataSetFactory:
    """Factory to create a data set input based on the configuration
    Attributes:
        game_factory (RobotRebootFactory): robot reboot factory to create samples for a dataset
        maze_size (int): size of the maze (number of rows and cols)
        max_depth (int): Max depth of the tree search
        cnn: neural network leading the mcts
        playouts: number of playouts per simulation
    """

    def __init__(self, maze_size, cnn, max_depth=20, seed=26, playouts=10):
        self.__game_factory = RobotRebootFactory(seed=seed)
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

    def create(self):
        """Creates a data set example by creating random games, searching on the tree to build the probabilities,
        executes a play of the game and obtains the value if the NN suggestions were followed
        Returns:
            v (int): value of the final game (win,loss or draw)
            p (np array): Probability distribution of wining per action based on the initial state of the game
            s (state): Initial state where the search and play where applied in its matrix form
        """
        game, state, robot_ids = self.game_factory.create(self.maze_size)
        model = RobotRebootModel(game, self.cnn)
        game_player = GamePlayer(model, game)
        mcts = MonteCarloTreeSearch(heuristic_fn, self.max_depth, game_player, playouts=self.playouts)
        p = mcts.search(state)
        final_state = game_player.play(state, self.max_depth)
        v = game.get_value(final_state)
        return v, p, state.get_matrix()