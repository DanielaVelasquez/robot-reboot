class UnsupportedMazeSize(Exception):
    def __init__(self, size):
        self.message = f"Factory does not have games for mazes with size {size}"
