import numpy as np

from src.agent.human import HumanAgent
from src.robot_reboot.factory import RobotRebootFactory
from src.robot_reboot.game import RobotRebootGame
from src.robot_reboot.goal_house import RobotRebootGoalHouse
from src.robot_reboot.state import RobotRebootState
from src.ui.robot_reboot_game_view import RobotRebootGameView
from src.view.base import view_play


def main():
    # np.random.seed(26)
    # factory = RobotRebootFactory()
    # game, game_state, selected_quadrants = factory.create(31, locate_robot_close_goal=True, max_movements=1)
    goal_house_pos = (0, 0)
    house = RobotRebootGoalHouse(1, goal_house_pos)
    maze = np.array([[0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0]
                     ])
    game = RobotRebootGame(4, maze, house)
    robot_1 = (2, 2)
    robot_2 = (4, 0)
    robot_3 = (4, 2)
    robot_4 = (4, 4)
    game_state = RobotRebootState(game, [robot_1, robot_2, robot_3, robot_4])
    print(game_state.robots_positions)
    print(game.goal_house)
    human_agent = HumanAgent()
    rrView = RobotRebootGameView(game)
    view_play(human_agent, game, game_state, rrView)


if __name__ == "__main__":
    main()
