from src.alphazero.game import Game
from src.robot_reboot.action import RobotRebootAction
from src.robot_reboot.state import RobotRebootState


class RobotRebootGoal:

    def __init__(self, robot, pos: tuple):
        self.robot = robot
        self.position = pos

    def __str__(self):
        return f'Robot {self.robot} on position {self.position}'


class RobotRebootGame(Game):
    def get_value(self, state: RobotRebootState):
        pass

    def is_over(self, state: RobotRebootState):
        pass

    def get_score(self, state: RobotRebootState):
        pass

    def apply(self, action: RobotRebootAction, state: RobotRebootState):
        pass