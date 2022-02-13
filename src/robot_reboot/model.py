import tensorflow as tf

from src.alphazero.model import Model
from src.encoders.maze_and_two_planes_per_robot import MazeAndTwoPlanesPerRobotEncoder
from src.exceptions.robot_reboot.model import NeuralNetworkOutputNotMatchingGameActions
from src.exceptions.util import assert_or_throw
from src.robot_reboot.game import RobotRebootGame
from src.robot_reboot.state import RobotRebootState


class RobotRebootModel(Model):
    def __init__(self, game: RobotRebootGame, network: tf.keras.Model, encoder=None):
        Model.__init__(self, 'Robot reboot model', game)
        assert_or_throw(network.outputs[1].shape[1] == len(game.actions), NeuralNetworkOutputNotMatchingGameActions())
        self.__network = network
        self.__encoder = encoder
        if self.__encoder is None:
            self.__encoder = MazeAndTwoPlanesPerRobotEncoder(game)

    @property
    def network(self):
        return self.__network

    def predict(self, state: RobotRebootState):
        s = self.__encoder.encode(state)
        rows, cols, layers = s.shape
        pred = self.__network.predict(s.reshape((1, rows, cols, layers)))
        return pred[0][0, 0], pred[1][0]

    def train(self, train_x, train_y, test_x, test_y):
        pass
