from src.alphazero.model import Model
from src.alphazero.state import State
from src.robot_reboot.game import RobotRebootGame


class UCTModel(Model):

    def __init__(self, game: RobotRebootGame):
        Model.__init__(self, 'Upper-Confidence bounds applied to trees model', game)

    def predict(self, state: State):
        pass

    def train(self, train_x, train_y, test_x, test_y):
        pass
