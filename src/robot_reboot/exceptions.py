class RequiredValueException(Exception):
    def __init__(self, value):
        self.message = f'{value} is required'
