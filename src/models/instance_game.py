import numpy as np
import queue
import json

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


def results_stats(results):
    with open(results) as f:
        results = json.load(f)
    games = results['games']
    i = 0
    total_movements = 0
    for g in games:
        print(f'Game {i}: ' + str(len(g['movements'])))
        total_movements += len(g['movements'])
        i += 1
    print(f'Average {total_movements / i}')


def load_game(results, index=0):
    with open(results) as f:
        results = json.load(f)
    robots = results['robots']
    game = results['games'][index]
    state_file = game['state_file']
    state = np.load(state_file)

    fake_goals = queue.Queue()

    for r in robots:
        fake_goals.put(Goal(r, (0, 0)))

    rr = RobotReboot(Maze(np.array([[0, 0, 0, 0, 0]])), fake_goals)
    movements_ = game['movements']
    rr.set_game(robots, state.astype(int), movements_)
    return rr, movements_
