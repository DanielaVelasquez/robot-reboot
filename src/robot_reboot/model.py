from src.alphazero.model import Model
from src.robot_reboot.game import RobotRebootGame
from src.robot_reboot.state import RobotRebootState


class RobotRebootModel(Model):
    def __init__(self, game: RobotRebootGame, network):
        Model.__init__(self, 'Robot reboot model', game)
        self.__network = network

    def predict(self, state: RobotRebootState):
        pass

    def train(self, train_x, train_y, test_x, test_y):
        pass
