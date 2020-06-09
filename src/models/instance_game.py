import numpy as np
import queue

from .robotreboot import RobotReboot
from .maze import Maze
from .robotreboot import Goal


def get_robot_reboot(seed=26):
    data = np.load('data/maze_v1.npy')  # .transpose()
    np.random.seed(seed)
    maze = Maze(data)
    goals = queue.Queue()
    goals.put(Goal("A", (5, 10)))
    goals.put(Goal("G", (4, 14)))
    goals.put(Goal("B", (9, 13)))
    goals.put(Goal("R", (1, 12)))
    goals.put(Goal("G", (14, 10)))
    goals.put(Goal("A", (9, 3)))
    goals.put(Goal("B", (11, 6)))
    goals.put(Goal("A", (3, 1)))
    goals.put(Goal("G", (12, 1)))
    goals.put(Goal("R", (5, 2)))
    goals.put(Goal("A", (11, 9)))
    goals.put(Goal("R", (13, 14)))
    goals.put(Goal("B", (3, 9)))
    goals.put(Goal("R", (14, 4)))
    goals.put(Goal("G", (4, 5)))
    goals.put(Goal("B", (1, 6)))

    robot_reboot = RobotReboot(maze, goals)
    return robot_reboot;
