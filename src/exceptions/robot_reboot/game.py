class NoRobotsGameException(Exception):
    def __init__(self):
        self.message = "Robots must be provided, it must be a positive number greater than zero"


class InvalidMazeException(Exception):
    def __init__(self):
        self.message = "Maze is invalid, check only walls and empty cells are display and that walls are depicted in " \
                       "the designated cells, i.e even rows and columns "


class RobotHouseOutOfMazeBoundsException(Exception):
    def __init__(self):
        self.message = "Robot's house is out of the maze bounds"


class RobotHouseInvalidRobotIdException(Exception):
    def __init__(self):
        self.message = "Robot id doesn't match the number of robots in the game"


class MazeNotSquareException(Exception):
    def __init__(self):
        self.message = "Maze must be squared"


class MazeSizeInvalidException(Exception):
    def __init__(self):
        self.message = "Maze's size must be odd"
