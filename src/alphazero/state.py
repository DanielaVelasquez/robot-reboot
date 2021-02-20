from abc import ABC, abstractmethod


class State(ABC):
    """ State of a game

    Attributes:
        sequence_id (int):  Moment in time where this state occurred
        game        (Game): Game that the state belongs to

    """

    def __init__(self, sequence_i, game):
        """ Initializes a state
        Args:
            sequence_i (int):  Moment in time where the state occurred i.e 0  it's how the game started
            game       (Game): Game that the state belongs to
        """
        self.__sequence_i = sequence_i
        self.__game = game

    @property
    def sequence_i(self):
        return self.__sequence_i

    @property
    def game(self):
        return self.game

    @abstractmethod
    def __str__(self):
        """" Creates a string representation of a state based on the game it belongs to
        """
        pass
