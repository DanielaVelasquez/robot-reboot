from src.alphazero.model import Model
from src.game.state import State
from src.robot_reboot.game import RobotRebootGame


class UctModel(Model):

    def __init__(self, game: RobotRebootGame):
        Model.__init__(self, 'Upper-Confidence bounds applied to trees model', game)

    def predict(self, state: State):
        return None, [1 for i in self.game.actions]
