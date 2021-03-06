import numpy as np

from exceptions.robot_reboot.factory import UnsupportedMazeSize
from src.robot_reboot.game import RobotRebootGame
from src.robot_reboot.game_configuration import RobotRebootGameConfiguration
from src.robot_reboot.goal_house import RobotRebootGoalHouse
from src.robot_reboot.maze_cell_type import MazeCellType
from src.robot_reboot.util import join_quadrants, transpose_position_to_quadrant


class RobotRebootFactory:

    def create(self, n):
        game_configurations, n_robots = self.__get_quadrants(n)
        configurations_size = len(game_configurations)
        assert configurations_size > 0, "At least one configuration is needed to create a game"
        index = np.arange(0, configurations_size)
        np.random.shuffle(index)
        c1 = game_configurations[index[0]]
        c2 = game_configurations[index[1 % configurations_size]]
        c3 = game_configurations[index[2 % configurations_size]]
        c4 = game_configurations[index[3 % configurations_size]]
        maze = join_quadrants(c1.quadrant, c2.quadrant, c3.quadrant, c4.quadrant)

        # Select a quadrant to put a house

        selected_quadrant_index = np.random.randint(0, high=4)
        conf = game_configurations[index[selected_quadrant_index % configurations_size]]
        houses = conf.houses
        selected_house_goal = houses[np.random.randint(0, high=len(houses))]
        x, y = transpose_position_to_quadrant(conf.quadrant, selected_house_goal.house, selected_quadrant_index + 1)

        goal = RobotRebootGoalHouse(selected_house_goal.robot_id, (x, y))
        state = None
        return RobotRebootGame(n_robots, maze, goal), state


    def __get_quadrants(self, n):
        if n == 11:
            quadrants = self.__get_5_x_5_quadrants()
        else:
            raise UnsupportedMazeSize(n)
        return quadrants

    def __build_quadrant(self, n, walls):
        maze = np.zeros((n, n), dtype=int)
        for x, y in walls:
            maze[x, y] = MazeCellType.WALL.value
        return maze

    def __get_5_x_5_quadrants(self):
        n = 5
        return [
                   RobotRebootGameConfiguration(
                       self.__build_quadrant(n, [(2, 1), (3, 2)]),
                       [RobotRebootGoalHouse(0, (2, 2))]
                   )
               ], 2
