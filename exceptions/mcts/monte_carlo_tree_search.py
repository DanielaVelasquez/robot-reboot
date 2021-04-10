class InvalidPlayoutException(Exception):
    def __init__(self):
        self.message = "playouts must be greater than 0"
