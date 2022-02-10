from abc import ABC, abstractmethod


class GameStateView(ABC):

    @abstractmethod
    def display(self, state):
        pass
