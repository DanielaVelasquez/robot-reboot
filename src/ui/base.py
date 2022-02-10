from abc import ABC, abstractmethod


class GameStateView(ABC):
    def __init__(self, game_state):
        self.__game_state = game_state

    @abstractmethod
    def update(self, game_state):
        pass
