from abc import ABC, abstractmethod

from exceptions.game.state import InvalidStateSequence
from exceptions.util import assertOrThrow


class State(ABC):
    """ State of a game

    Attributes:
        game        (Game): Game that the state belongs to
        sequence_id (int):  Moment in time where this state occurred

    """

    def __init__(self, game, sequence_i=0):
        """ Initializes a state
        Args:
            game       (Game): Game that the state belongs to
            sequence_i (int):  Moment in time where the state occurred i.e 0  it's how the game started
        """
        assertOrThrow(sequence_i >= 0, InvalidStateSequence())
        self.__game = game
        self.__sequence_i = sequence_i

    @property
    def sequence_i(self):
        return self.__sequence_i

    @property
    def game(self):
        return self.__game

    @abstractmethod
    def __str__(self):
        """" Creates a string representation of a state based on the game it belongs to
        """
        pass
