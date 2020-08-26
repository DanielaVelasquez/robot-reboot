import numpy as np
from enum import Enum


class Direction(Enum):
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4


class Maze:
    NORTH_WALL = 1
    EAST_WALL = 2
    SOUTH_WALL = 4
    WEST_WALL = 8
    EMPTY = 0

    WALLS = [NORTH_WALL, EAST_WALL, SOUTH_WALL, WEST_WALL]

    def __init__(self, cells: np.array):
        self.cells = cells
        self.size = cells.shape

    @property
    def height(self):
        return int(self.size[1])

    @property
    def width(self):
        return int(self.size[0])


def get_movement_same_direction_for_wall(wall):
    if wall == Maze.NORTH_WALL:
        return Direction.NORTH
    elif wall == Maze.SOUTH_WALL:
        return Direction.SOUTH
    elif wall == Maze.EAST_WALL:
        return Direction.EAST
    elif wall == Maze.WEST_WALL:
        return Direction.EAST
    else:
        raise Exception("Invalid wall value")


def get_wall_at_direction(direction: Direction):
    if direction == direction.NORTH:
        return Maze.NORTH_WALL
    elif direction == direction.SOUTH:
        return Maze.SOUTH_WALL
    elif direction == direction.WEST:
        return Maze.WEST_WALL
    elif direction == direction.EAST:
        return Maze.EAST_WALL


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
