class EmptyRobotsPositionException(Exception):
    def __init__(self):
        self.message = "Robots position must be provided"


class InvalidRobotsPositionException(Exception):
    def __init__(self):
        self.message = "Robot positions can be negative values"


class RobotsPositionOutOfMazeBoundsException(Exception):
    def __init__(self):
        self.message = "All robot positions must be inside of the maze bounds"


class NumberRobotsNotMatchingException(Exception):
    def __init__(self):
        self.message = "Number of robots in the game and the state does not match"


class InvalidRobotsList(Exception):
    def __init__(self):
        self.message = "A list with robots positions is expected"
