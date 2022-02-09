from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select_action(self, game_state):
        pass
