import numpy as np

from src.robot_reboot.direction import Direction
from src.robot_reboot.maze_cell_type import MazeCellType


def valid_maze(n_robots, maze):
    robot_cells = [i * 2 for i in range(int(n_robots / 2) + 1)]
    cells = maze[:, robot_cells][robot_cells, :]
    return np.all(cells == MazeCellType.EMPTY.value) and maze_only_walls_empty_cells(maze)


def maze_only_walls_empty_cells(maze):
    return np.all(np.logical_or(maze == MazeCellType.EMPTY.value, maze == MazeCellType.WALL.value))


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


def join_quadrants(q1, q2, q3, q4):
    """Join quadrants of a maze and adds a row/col full of zeros representing the empty wall between
    each quadrant
    Args:
        q1 (np array): upper left side of the maze
        q2 (np array): upper right side of the maze
        q3 (np array): down left side of the maze
        q4 (np array): down right side of the maze
    Returns:
         maze : maze composed by all given quadrants, joint by a wall row/col full of zeros
    """
    assert q1.shape == q2.shape == q3.shape == q4.shape, "All quadrants must have the same shape"
    rows, cols = q1.shape
    result = np.zeros((2 * rows + 1, 2 * cols + 1), dtype=int)
    result[0:rows, 0:cols] = q1
    result[0:rows, cols + 1:] = np.flip(q2, 1)
    result[rows + 1:, 0:cols] = np.flip(q3, 0)
    result[rows + 1:, cols + 1:] = np.flip(q4)
    return result


def transpose_position_to_quadrant(q, pos, target_q):
    """Calculates a position on the maze based on the relative position on the quadrant and the quadrant position
    Args:
        q (np array): quadrant where position is originally placed
        pos (tuple): position in the quadrant
        target_q (int): which quadrant it wants to transpose to

    Returns:
        x, y : position in the maze for the give position in a quadrant based on the quadrant location in the maze
    """
    x, y = pos
    rows, cols = q.shape
    if target_q == 2 or target_q == 4:
        y += (cols - y) * 2

    if target_q == 3 or target_q == 4:
        x += (rows - x) * 2

    return x, y


def build_matrix(n, cells, value=MazeCellType.WALL.value):
    """Builds a square array with walls in certain positions
    Args:
        n (int): matrix size 
        cells (list): list of tuples for positions where value must be set
        value (int): value for the specified cells
    """
    maze = np.zeros((n, n), dtype=int)
    for x, y in cells:
        maze[x, y] = value
    return maze


def generate_positions_except(n, size, pos):
    """Generate a set of different positions avoiding a position
    Args:
        n (int): number of positions to generate
        size (int): size of the values within the positions
        pos (tuple): position to avoid generating
    """
    positions = set()
    for i in range(n):
        rnd = np.random.randint(0, size, size=2)
        p = (rnd[0], rnd[1])
        while p == pos or p in positions:
            rnd = np.random.randint(0, size, size=2)
            p = (rnd[0], rnd[1])
        positions.add(p)
    return positions
