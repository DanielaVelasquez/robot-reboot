import numpy as np

from src.robot_reboot.game import RobotRebootGame
from src.robot_reboot.goal_house import RobotRebootGoalHouse


def get_robot_reboot_game(n_robots=4, maze_size=31):
    house = RobotRebootGoalHouse(0, (2, 2))
    maze = np.zeros((maze_size, maze_size))
    game = RobotRebootGame(n_robots, maze, house)
    return game
