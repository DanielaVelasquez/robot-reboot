import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from models.robotreboot import RobotReboot
from models.maze import Maze
from models.robotreboot import Goal

class RobotRebootEnv(gym.Env):
    ACTIONS = ["A_N", "A_S", "A_E", "A_W",
              "B_N", "B_S", "B_E", "B_W",
              "C_N", "C_S", "C_S", "C_W"]

    REWARD = 100
    PUNISHMENT = -1

    def __init__(self, robot_reboot):
        self.robot_reboot = robot_reboot
        self.reward_range = (RobotRebootEnv.PUNISHMENT, RobotRebootEnv.REWARD)
        # North, South, East or West for each robot
        self.action_space = spaces.Discrete(4 * self.robot_reboot.count_robots)

        # Observation is a three dimensional array
        obs = self.robot_reboot.state
        # low = np.zeros(len(obs_shape), dtype=int)
        # high = np.array(obs_shape, dtype=int) - np.ones(len(obs_shape), dtype=int)
        # self.observation_space = spaces.Box(low, high, dtype=int)
        self.observation_space = spaces.Box(0, RobotReboot.GOAL, shape=obs.shape, dtype=int)

        self.seed()
        self.reset()

    def step(self, action):
        robot_id, direction = action.split("_")
        self.robot_reboot.move_robot(robot_id, direction)
        done = self.robot_reboot.done
        if done:
            reward = RobotRebootEnv.REWARD
        else:
            reward = RobotRebootEnv.PUNISHMENT
        state = self.robot_reboot.state
        info = {}
        return state, reward, done, info

    def reset(self):
        self.robot_reboot.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == "__main__":
    data = np.load('./data/maze.npy')  # .transpose()
    np.random.seed(0)
    maze = Maze(data)
    goal = Goal("A", (0, 4))
    robots = {"A": (0, 0), "B": (4, 1), "C": (2, 2)}
    rr = RobotReboot(maze, robots, goal)
    env = RobotRebootEnv(rr)
