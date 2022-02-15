import numpy as np

from src.agent.human import HumanAgent
from src.robot_reboot.factory import RobotRebootFactory
from src.robot_reboot.game import RobotRebootGame
from src.robot_reboot.goal_house import RobotRebootGoalHouse
from src.robot_reboot.state import RobotRebootState
from src.ui.robot_reboot_game_view import RobotRebootGameView
from src.view.base import simulate_game


def main():
    np.random.seed(26)
    factory = RobotRebootFactory()
    game, game_state, selected_quadrants = factory.create(31, locate_robot_close_goal=True, max_movements=1)
    print(game_state.robots_positions)
    print(game.goal_house)
    human_agent = HumanAgent()
    rrView = RobotRebootGameView(game)
    simulate_game(human_agent, game, game_state, rrView)


if __name__ == "__main__":
    main()
