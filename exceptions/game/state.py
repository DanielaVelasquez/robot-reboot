class InvalidStateSequence(Exception):
    def __init__(self):
        self.message = "sequence must be a positive number"
