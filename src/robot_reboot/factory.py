import numpy as np

from exceptions.robot_reboot.factory import UnsupportedMazeSize
from src.robot_reboot.maze_cell_type import MazeCellType


def build_quadrant(n, walls):
    maze = np.zeros((n, n), dtype=int)
    for x, y in walls:
        maze[x, y] = MazeCellType.WALL.value
    return maze




class RobotRebootFactory:

    def create(self, n):
        quadrants = self.__get_quadrants(n)
        quadrants_size = len(quadrants)

        index = np.arange(0, quadrants_size)
        np.random.shuffle(index)
        q1 = quadrants[index[0 % quadrants_size]]
        q2 = quadrants[index[1 % quadrants_size]]
        q3 = quadrants[index[2 % quadrants_size]]
        q4 = quadrants[index[3 % quadrants_size]]

    def __get_quadrants(self, n):
        if n == 11:
            quadrants = self.__get_5_x_5_quadrants()
        else:
            raise UnsupportedMazeSize(n)
        return quadrants

    def __get_5_x_5_quadrants(self):
        return list()
