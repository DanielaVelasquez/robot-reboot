class InvalidDepthException(Exception):
    def __init__(self):
        self.message = "Tree depth must be greater than 0"


class InvalidPlayoutException(Exception):
    def __init__(self):
        self.message = "playouts must be greater than 0"
