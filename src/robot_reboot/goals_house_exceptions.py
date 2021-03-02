class InvalidRobotIdException(Exception):
    def __init__(self):
        self.message = "Robot id must be a positive number"
