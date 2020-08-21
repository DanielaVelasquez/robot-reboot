import queue

from abc import ABC, abstractmethod


class Game(ABC):

    def __init__(self):
        self.actions = queue.LifoQueue()

    @abstractmethod
    def score(self):
        """ Game score

        Returns:
            z : -1 if is a loss, 0 if is a drawn or nothing happened, 1 if is a win
        """
        pass

    @abstractmethod
    def is_over(self):
        """Determines whether the game is over

        Returns:
            bool: True to indicate the game is over. False otherwise.
        """
        pass

    @abstractmethod
    def valid_actions(self, state):
        """Gets valid actions based on the current game state

        Args:
             state: Current game state

        Returns:
            list: List with all the valid actions from current state
        """
        pass

    def move(self, action):
        """Executes a move on the game and record the action on a queue

        Args:
            action: action to execute
        """
        self.actions.put(action)
        self.move()

    @abstractmethod
    def __execute_move(self, action):
        """Executes a move on the game

        Args:
            action: action to execute
        """
        pass

    def undo_move(self):
        """Undo the last action executed on the gamr and remove it from the queue
        """
        last_action = self.actions.get()
        self.__execute_undo_move(last_action)

    @abstractmethod
    def __execute_undo_move(self, action):
        """Returns the game to the previous state before the action was executed

        Args:
            action: last action executed on the game
        """
        pass
