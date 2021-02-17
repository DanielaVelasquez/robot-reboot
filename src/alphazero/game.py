from abc import ABC, abstractmethod
from src.alphazero.state import State
from src.alphazero.action import Action


class Game(ABC):
    """" Represents the rules of a game with all its possible actions and it's capable
    of interpreting what a state mean in terms of the game

    Attributes:
        actions (list): available actions for the game that can be applied to a state of the game

    """

    def __init__(self, actions):
        """" Initializes a game
        Args:
            actions (list): list of valid actions for the game
        """
        self.__actions = actions

    @property
    def actions(self):
        return self.__actions

    @abstractmethod
    def get_value(self, state: State):
        """ Gets the value for a state
        Args:
            state (State): state to evaluate the the game on
        Returns:
            1 if it is a win
            -1 if it is lost
            0 if it is a draw
        """
        pass

    @abstractmethod
    def is_over(self, state: State):
        """" Determines if a game is over based on a state
            Args:
                state (State): state to evaluate if the game is over
        """
        pass

    @abstractmethod
    def get_score(self, state: State):
        """" Calculates the score for a state
        Args:
            state (State): state to calculate the score on
        """
        pass

    @abstractmethod
    def apply(self, action: Action, state: State):
        """" Applies an action over a state to get to the next state
        Args:
            action (Action): action to perform on the state
            state (State):  state to apply to the action
        Returns:
            resulting state from applying action on the give state
        """
        pass
