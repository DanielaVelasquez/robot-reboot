class EdgeNotVisitedException(Exception):
    def __init__(self, action_i):
        self.message = f"No visits have been registered for action: {action_i}"


class InvalidNumberActionsException(Exception):
    def __init__(self):
        self.message = "Number of actions must be above zero"


class InvalidActionsTypeException(Exception):
    def __init__(self):
        self.message = "Number of actions must be an integer"
