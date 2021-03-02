import numpy as np

from src.robot_reboot.direction import Direction
from src.robot_reboot.maze_cell_type import MazeCellType


def valid_maze(n_robots, maze):
    robot_cells = [i * 2 for i in range(int(n_robots / 2) + 1)]
    cells = maze[:, robot_cells][robot_cells, :]
    return np.all(cells == MazeCellType.EMPTY.value) and maze_only_walls_empty_cells(maze)


def maze_only_walls_empty_cells(maze):
    return np.all(np.logical_or(maze == MazeCellType.EMPTY.value,maze == MazeCellType.WALL.value))




def get_opposite_direction(movement: Direction):
    if movement == Direction.NORTH:
        return Direction.SOUTH
    elif movement == Direction.SOUTH:
        return Direction.NORTH
    elif movement == Direction.EAST:
        return Direction.WEST
    elif movement == Direction.WEST:
        return Direction.EAST
    else:
        raise Exception("Invalid movement")


def get_cell_at(direction: Direction, position: tuple, max_rows, max_cols):
    """Returns immediate cell in the specified direction. For instance if I request cell at north from position
    (1,1), cell (0, 1) will be the answer.

    Args:
        direction: point of reference to know which cell should be retrieved
        position: starting point
        max_rows: max number of rows
        max_cols: max number of columns

    Return:
        tuple: with cell on the speficied position from starting point.

    Raises:
        Exception: if there is not available cell in that direction, this can occur
                    when the starting point is at a border
    """
    x, y = position
    if direction == Direction.NORTH and x != 0:
        return x - 1, y
    elif direction == Direction.SOUTH and x != max_rows - 1:
        return x + 1, y
    elif direction == Direction.WEST and y != 0:
        return x, y - 1
    elif direction == Direction.EAST and y != max_cols - 1:
        return x, y + 1
    else:
        raise Exception("Not cell available")


