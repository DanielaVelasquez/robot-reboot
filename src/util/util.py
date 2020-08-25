from src.alphazero.robot_reboot_game import Maze, RobotRebootGame
from enum import Enum


class Direction(Enum):
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4


def get_movement_same_direction_for_wall(wall):
    if wall == Maze.NORTH_WALL:
        return RobotRebootGame.NORTH_MOVE
    elif wall == Maze.SOUTH_WALL:
        return RobotRebootGame.SOUTH_MOVE
    elif wall == Maze.EAST_WALL:
        return RobotRebootGame.EAST_MOVE
    elif wall == Maze.WEST_WALL:
        return RobotRebootGame.WEST_MOVE
    else:
        raise Exception("Invalid wall value")


def get_opposite_direction(movement):
    if movement == RobotRebootGame.NORTH_MOVE:
        return RobotRebootGame.SOUTH_MOVE
    elif movement == RobotRebootGame.SOUTH_MOVE:
        return RobotRebootGame.NORTH_MOVE
    elif movement == RobotRebootGame.EAST_MOVE:
        return RobotRebootGame.WEST_MOVE
    elif movement == RobotRebootGame.WEST_MOVE:
        return RobotRebootGame.EAST_MOVE
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
