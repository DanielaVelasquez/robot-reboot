class EmptyRobotsPositionException(Exception):
    def __init__(self):
        self.message = "Robots position must be provided"


class InvalidRobotsPositionException(Exception):
    def __init__(self):
        self.message = "Robot positions can be negative values"


class RobotsPositionOutOfMazeBoundsException(Exception):
    def __init__(self):
        self.message = "All robot positions must be inside of the maze bounds"
