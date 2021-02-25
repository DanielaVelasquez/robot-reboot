import gym
from gym import spaces
from gym.utils import seeding
from models.instance_game import get_robot_reboot
from models.robotreboot import RobotReboot


class RobotRebootEnv(gym.Env):
    DONE_REWARD = 100
    MOVING_GOAL_ROBOT_REWARD = 2
    MOVEMENT_PUNISHMENT = -1
    SAME_STATE_PUNISHMENT = -10

    def __init__(self, robot_reboot:RobotReboot):
        self.robot_reboot = robot_reboot
        self.reward_range = (RobotRebootEnv.MOVEMENT_PUNISHMENT, RobotRebootEnv.DONE_REWARD)
        # North, South, East or West for each robot
        self.action_space = spaces.Discrete(len(self.actions))

        # Observation is a three dimensional array
        obs = self.robot_reboot.state(normalize=True)
        # low = np.zeros(len(obs_shape), dtype=int)
        # high = np.array(obs_shape, dtype=int) - np.ones(len(obs_shape), dtype=int)
        # self.observation_space = spaces.Box(low, high, dtype=int)
        self.observation_space = spaces.Box(0, RobotReboot.GOAL, shape=obs.shape, dtype=float)

        self.seed()
        self.reset()

    def step(self, action):
        robot_id, direction = action.split("_")
        prev_state = self.robot_reboot.state(normalize=True)
        self.robot_reboot.move_robot(robot_id, direction)
        state = self.robot_reboot.state(normalize=True)

        done = self.robot_reboot.done
        reward = 0
        if done:
            reward += RobotRebootEnv.DONE_REWARD
        else:
            if (prev_state == state).all():
                reward += RobotRebootEnv.SAME_STATE_PUNISHMENT
            elif robot_id == self.robot_reboot.goal_house.robot_id:
                reward += RobotRebootEnv.MOVING_GOAL_ROBOT_REWARD
            reward += RobotRebootEnv.MOVEMENT_PUNISHMENT
        info = {}
        return state, reward, done, info

    def reset(self):
        self.robot_reboot.next_round()
        return self.robot_reboot.state(normalize=True)

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
        return self.robot_reboot.state(normalize=True)


if __name__ == "__main__":
    robot_reboot = get_robot_reboot()
    print(robot_reboot.state)
    env = RobotRebootEnv(robot_reboot)
