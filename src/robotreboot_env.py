import gym
from gym import spaces
from gym.utils import seeding
from models.instance_game import get_robot_reboot
from models.robotreboot import RobotReboot


class RobotRebootEnv(gym.Env):
    REWARD = 100
    PUNISHMENT = -1

    def __init__(self, robot_reboot):
        self.robot_reboot = robot_reboot
        self.reward_range = (RobotRebootEnv.PUNISHMENT, RobotRebootEnv.REWARD)
        # North, South, East or West for each robot
        self.action_space = spaces.Discrete(len(self.actions))

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
        self.robot_reboot.next_round()
        return self.robot_reboot.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def actions(self):
        actions = list()
        for robot_id in self.robot_reboot.robots:
            actions.append(f'{robot_id}_N')
            actions.append(f'{robot_id}_S')
            actions.append(f'{robot_id}_E')
            actions.append(f'{robot_id}_W')
        return actions

    @property
    def state(self):
        return self.robot_reboot.state


if __name__ == "__main__":
    robot_reboot = get_robot_reboot()
    print(robot_reboot.state)
    env = RobotRebootEnv(robot_reboot)
