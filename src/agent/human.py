from src.agent.base import Agent
from src.robot_reboot.action import RobotRebootAction
from src.robot_reboot.direction import Direction


def action_from_str(value):
    robot_id = value[0]
    d = value[1]
    if d == 'N':
        direction = Direction.North
    elif d == 'S':
        direction = Direction.South
    elif d == 'E':
        direction = Direction.East
    elif d == 'W':
        direction = Direction.West

    return RobotRebootAction(int(robot_id), direction)


class HumanAgent(Agent):
    def select_action(self, game_state):
        human_move = input('-- ')
        return action_from_str(human_move)
