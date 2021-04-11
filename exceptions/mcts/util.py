class InvalidDepthException(Exception):
    def __init__(self):
        self.message = "Tree depth must be greater than 0"
